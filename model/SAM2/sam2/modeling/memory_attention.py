# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import copy

import torch
from torch import nn, Tensor

from .sam.transformer import RoPEAttention

from .sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,  ## 256
        dim_feedforward: int,  ## 2048
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        local_branch: bool = False,  ## 3/27 Experiment
        local_cross_attention: nn.Module = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        self.local_branch = local_branch
        if self.local_branch:
            self.norm2_lc = None
            if local_cross_attention is None:
                self.cross_attn_image_local = copy.deepcopy(self.cross_attn_image)
            else:
                self.cross_attn_image_local = local_cross_attention
                self.norm2_lc = copy.deepcopy(self.norm2)  ###

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0, memory_local=None, memory_pos_local=None):
        """
        tgt: [B, H*W, C_q]
        memory: [B, H*W*M, C_kv]
        """
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        # print(f'[SAM2 global_branch] memory {memory.shape} kwds{kwds}')
        tgt2_global = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2_global)

        ## local memory cross attention
        if self.local_branch:
            assert memory_local is not None
            if self.norm2_lc is not None:
                tgt2_lc = self.norm2(tgt)
            else:
                tgt2_lc = tgt2
            # print(f'[SAM2 local_branch] memory_local {memory_local.shape} kwds{kwds}')
            tgt2_local = self.cross_attn_image_local(
                q=tgt2_lc + query_pos if self.pos_enc_at_cross_attn_queries else tgt2_lc,
                k=memory_local + memory_pos_local if self.pos_enc_at_cross_attn_keys else memory_local,
                v=memory_local,
                **kwds,
            )
            tgt = tgt + self.dropout2(tgt2_local)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        memory_local: Optional[Tensor] = None,
        memory_pos_local: Optional[Tensor] = None,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        if self.self_attn is not None:
            tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope, memory_local, memory_pos_local)
        # MLP
        tgt2 = self.norm3(tgt)  ## [B, hw, 256]
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
        input_norm: bool = False,  ## 1/9 Experiment: norm on input
        dense_skip: bool = False,   ## 1/20 Experiment: like unet++
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.curr_norm, self.mem_norm = None, None
        if input_norm:
            self.curr_norm = nn.LayerNorm(d_model) ## nn.InstanceNorm2d(d_model)
            self.mem_norm = nn.LayerNorm(64) ## nn.InstanceNorm2d(64)
        self.dense_skip = dense_skip

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        memory_local: Optional[Tensor] = None,
        memory_pos_local: Optional[Tensor] = None,
    ):
        """
        Input [HW,B,C] will be convert to [B,HW,C]
        """
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )
        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.curr_norm is not None:
            output = self.curr_norm(output)
        if self.mem_norm is not None:
            memory = self.mem_norm(memory)
            if memory_local is not None:
                memory_local = self.mem_norm(memory_local)
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
            if memory_local is not None:
                memory_local = memory_local.transpose(0, 1)
                memory_pos_local = memory_pos_local.transpose(0, 1)

        output_list = []
        for i in range(self.num_layers):
            ## 12/20 Experiment: dense skip connection (not used now)
            if self.dense_skip:
                if len(output_list) > 0:
                    for past_output in output_list:
                        output = output + past_output
                output_list.append(output)

            layer = self.layers[i]
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            # print(f'[mematt] memory {memory.dtype} memory_pos {memory_pos.dtype} memory_local {memory_local.dtype} memory_pos_local {memory_pos_local.dtype}')
            output = layer(
                tgt=output,    ## [hw, B, 256]
                memory=memory, ## [hw*M, B, 64] hw=24*24=576
                pos=memory_pos,
                query_pos=curr_pos,
                memory_local=memory_local,
                memory_pos_local=memory_pos_local,
                **kwds,
            ) ## output already include residual in att layer

        normed_output = self.norm(output)
        del output_list

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        return normed_output  ## [hw, B, 256]
