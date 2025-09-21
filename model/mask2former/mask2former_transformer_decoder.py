# Mask2Former Transformer Decoder
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional

try:
    from ...archive.deAOT import GatedPropagation, DWConv1d, silu
except:
    pass

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# We only use one needle query in the Mask2Former decoder!
# However, if we want to use the other query to decode the fascia, we can add the following code.
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False, rank = 0, block="crossatt"):
        super().__init__()
        # if rank > 0:
        #     self.multihead_attn = LoRAMultiheadAttention(d_model, nhead, dropout=dropout, r = rank)
        # else:
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.block = block
        if 'gate' in self.block:
            self.gate = nn.ModuleDict({'conv': DWConv1d(d_model),
                                       'linear': nn.Linear(d_model, d_model)}) 
            self.linear_U = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(  ## default
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        if 'gate' in self.block:
            l, bs, _ = tgt.size()
            _, bs, hidden_dim = memory.size()
            u = self.get_U(tgt)
            tgt = tgt.reshape(l, bs, -1) * u
            tgt = self.gate['conv'](tgt, hidden_dim)
            tgt = self.gate['linear'](tgt)

        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)

    def set_dropout_rate(self, rate):
        """Dynamically set dropout rate"""
        self.dropout = nn.Dropout(rate)

    def get_U(self, tgt):  ## from GatedPropagationModule
        """
        Norm (is done in MultiScaleMaskedTransformerDecoder.decoder_norm), linear, activation
        """
        curr_U = self.linear_U(tgt)  ## -> d * expand_ratio
        cat_curr_U = silu(curr_U)
        return cat_curr_U

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False, rank=0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # if rank > 0:
        #     self.linear1 = LoRALinear(d_model, dim_feedforward)
        #     self.linear2 = LoRALinear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
    
    def set_dropout_rate(self, rate):
        """Dynamically set dropout rate"""
        self.dropout = nn.Dropout(rate)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# In the following code, we remove all the self-attention layers, the query positional encoding and the classification head
# since there's only one single class: the needle.
# Nevertheless, the effect of the classification head has not been tested yet. Can try.
class MultiScaleMaskedTransformerDecoder(nn.Module):

    def __init__(
        self,
        in_channels=[144, 288, 576],
        mask_classification=False,
        det_head=False,
        # *,
        num_classes=1,
        hidden_dim=256,
        num_queries=1,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=1,
        pre_norm=False,
        mask_dim=72,
        frame_query_layers=3,
        image_size=384,
        ca_block = 'crossatt',
        # enforce_input_project: bool,
        lora_rank = 0,
        vis_query_att = False,  ## 2/7 Experiment
        branch = 1, ## 3/24 Experiment
        short_term_t = 1,
        qatt_block = 'crossatt',  ## options: 'crossatt' 'gp'
        query_membank_att = False,  ## 3/3 Experiment
        moe_q = None,
        ens_q = False ## 3/17 Experiment
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
        """
        super().__init__()

        # assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.det_head = det_head
        self.image_size = image_size

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        # self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        ## Utilize output query by VIS modules
        self.vis_query_att = vis_query_att
        if vis_query_att:
            self.gated_prop, self.use_mem = True, True
            self.vis_query_att_layers = nn.ModuleList()
            self.branch, self.short_term_t, self.qatt_block = branch, short_term_t, qatt_block
            if self.branch == 2:
                self.vis_query_att_short_term_layers = nn.ModuleList()
                self.dual_branch_LN_layers = nn.ModuleList()
            self.fq_pos = nn.Embedding(num_queries, hidden_dim)  ## num_frame_queries
            if self.gated_prop and self.qatt_block == 'crossatt': ## Gratt VIS
                from ..vis.gumble_softmax import GumbelSoftmax
                ## W_gate
                self.prop_instance = nn.Linear(hidden_dim, 1)  ## not used in qatt7
                # self.prop_instance = MLP(hidden_dim, hidden_dim, 1, 3) ## no use
                #self.threshold = nn.Parameter(torch.tensor(0.5))
                self.gumble_gate = GumbelSoftmax(eps=0.66667)  ## softmax temperature
            if self.use_mem: ## Gen VIS
                # self.out_query_norm = nn.LayerNorm(hidden_dim)  ## original self.decoder_norm in grattvisdecoder
                self.pre_memory_embed_k = nn.Linear(hidden_dim, hidden_dim)
                self.pre_memory_embed_v = nn.Linear(hidden_dim, hidden_dim)
                self.pre_query_embed_k = nn.Linear(hidden_dim, hidden_dim)
                self.pre_query_embed_v = nn.Linear(hidden_dim, hidden_dim)  ## no need in qatt3 & qatt4

        self.query_membank_att = query_membank_att
        if query_membank_att:
            from ..SAM2.sam2.modeling.sam.transformer import RoPEAttention
            self.query_membank_att_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            # self.transformer_self_attention_layers.append(
            #     SelfAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.0,
            #         normalize_before=pre_norm,
            #     )
            # )
            if vis_query_att: ## no need in qatt4
                if self.qatt_block == 'crossatt':
                    self.vis_query_att_layers.append(
                        CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )
                elif self.qatt_block == 'gp':
                    att_nhead=2
                    d_att = hidden_dim // 2 if att_nhead == 1 else hidden_dim // att_nhead
                    self.vis_query_att_layers.append(
                        GatedPropagation(d_qk=hidden_dim,
                                    d_vu=hidden_dim, # origin: * 2
                                    num_head=att_nhead,
                                    use_linear=False,
                                    dropout=0.,
                                    d_att=d_att,
                                    top_k=-1,
                                    expand_ratio=1., # origin: 2. when c_k = 128 and cat_curr_U dim is c_v*2
                                    dim_2D=False)
                    )
                if self.branch == 2:  ## short term branch concept in DeAOT
                    if self.qatt_block == 'crossatt':
                        self.vis_query_att_short_term_layers.append(
                            CrossAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )
                    elif self.qatt_block == 'gp':
                        self.vis_query_att_short_term_layers.append(
                            GatedPropagation(d_qk=hidden_dim,
                                        d_vu=hidden_dim, # origin: * 2
                                        num_head=att_nhead,
                                        use_linear=False,
                                        dropout=0.,
                                        d_att=d_att,
                                        top_k=-1,
                                        expand_ratio=1.,
                                        dim_2D=False)
                        )
                    self.dual_branch_LN_layers.append(nn.LayerNorm(hidden_dim))
            if query_membank_att:
                self.query_membank_att_layers.append(
                    # CrossAttentionLayer(
                    #     d_model=hidden_dim,
                    #     nhead=nheads//4,
                    #     dropout=0.0,
                    #     normalize_before=pre_norm,
                    # )
                    RoPEAttention(
                        rope_theta=10000.0,
                        feat_sizes=[self.image_size//16, self.image_size//16],
                        rope_k_repeat=True,
                        embedding_dim=hidden_dim,
                        num_heads=1, #nheads//4,
                        downsample_rate=1,
                        dropout=0.1,
                        kv_in_dim=64,
                    )
                )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    rank = lora_rank,
                    block = ca_block
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    rank = lora_rank
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.frame_query_layers = frame_query_layers

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query position embedding (removed because we only have one query)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        # https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py#L324
        for i in range(self.num_feature_levels):
            # if isinstance(in_channels, int) and in_channels == hidden_dim:
            if in_channels[2 - i] == hidden_dim:
                self.input_proj.append(nn.Sequential())
            else:
                # here we enforce input projection to hidden dim
                self.input_proj.append(nn.Conv2d(in_channels[2 - i], hidden_dim, kernel_size=1))
                # weight_init.c2_xavier_fill(self.input_proj[-1])  ## not sure why but sometimes cause bottleneck in ipynb

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        if self.det_head:
            self.line_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        

    def forward(self, x, mask_features, mask=None, pre_memory=None):
        """
        Input:
            x: list of multi-scale feature
            mask_features: [B, C=72, H, W]
        Return:
            out: dict() of prediction mask
            instance_embed: [B, fQ, hidden_dim=256]
            frame queries: work as KV in Grattvism, [L, B, fQ, hidden_dim=256]
        Symbols:
            L: Number of Layers.
            B: Batch size.
            fQ: Number of frame-wise queries from Mask2Former.
            E: hidden dim
        """
        # assert len(x) == self.num_feature_levels
        x_copy = x.copy()
        if len(x) > self.num_feature_levels:
            x = x[:self.num_feature_levels]
        src = []
        pos = []
        size_list = []
        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):  ## channel must all be 256 if pixel decoder
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])  ## x: B,emb_dim,H',W' -> input_proj flat -> B,hidden_dim,H'W'

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, hidden_dim = src[0].shape

        # QxNxC
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # [fQ, B, hidden_dim]
        
        predictions_class = []
        predictions_cals = [] 
        predictions_mask = []
        if not self.mask_classification:
            predictions_class = [None]
        if not self.det_head:
            predictions_cals = [None]

        ## VITA paper: use the outputs from the last 3 layers for training VITA to save computation cost.
        query1 = self.decoder_norm(output)
        query1 = query1.transpose(0, 1)
        frame_queries = [query1]
        
        # prediction heads on learnable query features
        outputs_class, outputs_mask, outputs_cals, attn_mask, frame_query = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        
        predictions_mask.append(outputs_mask)
        if self.mask_classification:
            predictions_class.append(outputs_class)
        if self.det_head:
            predictions_cals.append(outputs_cals)

        # ## Experiment: vis_query_att (qatt2)
        if (pre_memory is not None and (self.vis_query_att and self.use_mem)):
            # pre query embed
            pre_query_k = self.pre_query_embed_k(output) # cQ, LB, E  (1,1,256)
            pre_query_v = self.pre_query_embed_v(output) # cQ, LB, E

            # pre memory read
            pre_memory_k  = pre_memory.get("k",[])
            pre_memory_v  = pre_memory.get("v",[])
            if len(pre_memory_k) > 0 and len(pre_memory_v) > 0:
                if self.branch == 2:
                    if len(pre_memory_k) > self.short_term_t:
                        pre_memory_k_st = torch.cat(pre_memory_k[-self.short_term_t:]).flatten(1,2) # Ms, LB, cQ, E
                        pre_memory_v_st = torch.cat(pre_memory_v[-self.short_term_t:]).flatten(1,2)
                    else:
                        pre_memory_k_st = torch.cat(pre_memory_k).flatten(1,2) # M, LB, cQ, E
                        pre_memory_v_st = torch.cat(pre_memory_v).flatten(1,2)
                pre_memory_k = torch.cat(pre_memory_k).flatten(1,2) # M, LB, cQ, E
                pre_memory_v = torch.cat(pre_memory_v).flatten(1,2) # M, LB, cQ, E (M,1,1,256)
                # print(f"[pre_memory_k] {pre_memory_k.shape} [pre_memory_v] {pre_memory_k.shape}")
            else:
                if self.branch == 2:
                    pre_memory_k_st = torch.empty((0, bs*1, 1, hidden_dim), device=output.device)
                    pre_memory_v_st = torch.empty((0, bs*1, 1, hidden_dim), device=output.device)
                pre_memory_k = torch.empty((0, bs*1, 1, hidden_dim), device=output.device)
                pre_memory_v = torch.empty((0, bs*1, 1, hidden_dim), device=output.device)
            
            if self.qatt_block != "gp":
                qk_mk = torch.einsum("qbc, mbpc -> bqmp", pre_query_k, pre_memory_k) # LB, cQ, M, cQ (similarity between memory)
                qk_mk = torch.einsum("bqmq -> bqm", qk_mk) # LB, cQ, M (Q=1, so this is only squeeze to remove one of the Qs)
                qk_mk = F.softmax(qk_mk, dim=2)
                qk_mk_mv = torch.einsum("bqm, mbqc-> qbc", qk_mk, pre_memory_v) # cQ, B, E

                mem_query = pre_query_v + qk_mk_mv  # cQ, LB, E  ## Experiment qatt3: remove pre_query_v +
                # output_q = output_q + pre_query_v        # cQ, LB, E  ## update

                if self.branch == 2:
                    qk_mk_st = torch.einsum("qbc, mbpc -> bqmp", pre_query_k, pre_memory_k_st) # LB, cQ, M, cQ (similarity between memory)
                    qk_mk_st = torch.einsum("bqmq -> bqm", qk_mk_st) # LB, cQ, M (Q=1, so this is only squeeze to remove one of the Qs)
                    qk_mk_st = F.softmax(qk_mk_st, dim=2)
                    qk_mk_mv_st = torch.einsum("bqm, mbqc-> qbc", qk_mk_st, pre_memory_v_st) # cQ, B, E
                    mem_query_st = pre_query_v + qk_mk_mv_st
        else:
            mem_query = None

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,            ## [fQ, B, hidden_dim]
                src[level_index],  ## [H'W', B, hidden_dim]
                memory_mask=attn_mask,  ## this is the "masked attention"
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=None,  # we do not apply positional encoding on query
            )

            # output = self.transformer_self_attention_layers[i](
            #     output, tgt_mask=None,
            #     tgt_key_padding_mask=None,
            #     query_pos=query_embed
            # )

            if self.vis_query_att and pre_memory is not None and self.use_mem:  ## between cross&ffn
                dec_pos = self.fq_pos.weight[None, :, None, :].repeat(1, 1, 1*bs, 1).flatten(0, 1) # fQ, LB, E
                if self.branch == 2:
                    if self.qatt_block == "crossatt":
                        st_output = self.vis_query_att_short_term_layers[i](
                            output, ## q
                            mem_query_st,    ## k,v
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=dec_pos, query_pos=None  #query_embed
                            )
                        ## qatt5
                        if self.gated_prop:  # check if instace is there in current frame, then propagate else not, only take last layer (frm M2F) op
                            _, st_output = self.gratt_module(bs, 1, mem_query, st_output, pred_prop_soft_list=[])
                    elif self.qatt_block == "gp":
                        cat_curr_U = self.vis_query_att_short_term_layers[i].get_U(output)
                        st_output, _ = self.vis_query_att_short_term_layers[i](
                            output, ## q
                            pre_memory_k_st.permute(0,2,1,3).flatten(0,1),   ## k [M,B,Q,E]->[MQ,B,E]
                            pre_memory_v_st.permute(0,2,1,3).flatten(0,1),   ## v
                            cat_curr_U, size_2d = hidden_dim
                            )
                if self.qatt_block == "crossatt":
                    output = self.vis_query_att_layers[i](
                        output, ## q
                        mem_query,    ## k,v
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=dec_pos, query_pos=None  #query_embed
                    )
                    ## qatt5
                    if self.gated_prop:  # check if instace is there in current frame, then propagate else not, only take last layer (frm M2F) op
                        _, output = self.gratt_module(bs, 1, mem_query, output, pred_prop_soft_list=[])
                elif self.qatt_block == "gp":
                    cat_curr_U = self.vis_query_att_layers[i].get_U(output)
                    lt_output, _ = self.vis_query_att_layers[i](
                        output, ## q
                        pre_memory_k.permute(0,2,1,3).flatten(0,1),   ## k [M,B,Q,E]->[MQ,B,E]
                        pre_memory_v.permute(0,2,1,3).flatten(0,1),   ## v
                        cat_curr_U, size_2d = hidden_dim
                        )
                    if self.branch == 1:
                        output =  output + lt_output
                if self.branch == 2:
                    if self.qatt_block == "gp":
                        output =  output + lt_output + st_output 
                    else:   
                        output =  output + st_output
                    output = self.dual_branch_LN_layers[i](output)

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, outputs_cals, attn_mask, frame_query = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )

            frame_queries.append(frame_query)
            predictions_mask.append(outputs_mask)
            if self.mask_classification:
                predictions_class.append(outputs_class)
            if self.det_head:
                predictions_cals.append(outputs_cals)

        if self.mask_classification:
            assert len(predictions_class) == self.num_layers + 1

        # the final output instance query feature before fed into MLP for mask prediction
        # this will be used for video instance segmentation
        # instance_embed = self.decoder_norm(output).transpose(0, 1)  # [fQ, B, hidden_dim] -> [B, fQ, hidden_dim]
        instance_embed = frame_queries[-1]

        out = {
            "pred_class": predictions_class[-1],
            "pred_cals": predictions_cals[-1],
            "pred_masks": predictions_mask[-1],  ## [B, fQ, H, W]
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, predictions_cals),
            "instance_embed": instance_embed,
        }

        
        if (self.vis_query_att and self.use_mem):  #  or (self.moe_q is not None and 'past' in self.moe_q)
            memory_input = frame_query.view(1, bs, self.num_queries, hidden_dim)  ## B,1,256
            pre_memory_k = self.pre_memory_embed_k(memory_input)[None] # 1, L, B, cQ, E
            pre_memory_v = self.pre_memory_embed_v(memory_input)[None] # 1, L, B, cQ, E ([None] adds a first dim)
            out['pre_memory'] = {"k": pre_memory_k, "v": pre_memory_v}
            ## TODO pred_prop_soft

        ## https://github.com/sukjunhwang/VITA/blob/main/vita/modeling/transformer_decoder/vita_mask2former_transformer_decoder.py#L437
        frame_queries = torch.stack(frame_queries[-self.frame_query_layers:]) # L x BT x fQ x 256
        # print("td pred_masks " , out['pred_masks'].shape, "frame q ",frame_queries.shape)
        return out, frame_queries

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        if self.mask_classification:
            outputs_class = self.class_embed(decoder_output)
            outputs_class = outputs_class.sigmoid()
        else:
            outputs_class = None
        if self.det_head:
            outputs_cals = self.line_embed(decoder_output)
            outputs_cals = outputs_cals.sigmoid()
        else:
            outputs_cals = None

        mask_embed = self.mask_embed(decoder_output)
        # q: num_queries, c: mask_dim
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        # apply sigmoid to the output masks
        outputs_mask = outputs_mask.sigmoid()

        return outputs_class, outputs_mask, outputs_cals, attn_mask, decoder_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_cals):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            if self.det_head:
                return [
                    {"pred_class": a, "pred_masks": b, "pred_cals": c} for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_cals[:-1])
                ]
            else:
                return [
                    {"pred_class": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


    ## Gratt-VIS
    ## Gating
    def gratt_module(self, B, L, incoming_output, output, pred_prop_soft_list):
        dec_out = self.decoder_norm(output)  # cQ, LB, E
        ## decide whether to propogate queries through frames
        proj_output = self.prop_instance(dec_out)
        # print(f'[dec_out] {dec_out.shape} [self.prop_instance(dec_out)]', proj_output.shape)
        pred_prop_hard, pred_prop_soft = self.gumble_gate(
            proj_output)  # [-1]  # take final layer output
        # it takes the gating output that attend the mask2former final layer
        pred_prop_hard = pred_prop_hard.transpose(0, 1).view(L, B, self.num_queries, 1)[-1]  # .repeat(L, 1, 1).transpose(0,1)  # take the final hard pred
        if self.training:
            pred_prop_soft_list.append(pred_prop_soft.transpose(0, 1).view(L, B, self.num_queries, 1))
        attn_mask = torch.zeros([B, self.num_queries, self.num_queries]).to(output)
        mask_idx = pred_prop_hard[:, :, 0] == 0
        attn_mask[mask_idx] = 1
        for b in range(attn_mask.size(0)):
            attn_mask[b].fill_diagonal_(0)
        attn_mask = attn_mask[None, :, None].repeat(L, 1, self.num_heads, 1, 1).view(L * B * self.num_heads, self.num_queries, self.num_queries)
        pred_prop_hard = pred_prop_hard.repeat(L, 1, 1).transpose(0, 1)
        output = pred_prop_hard * output + (1 - pred_prop_hard) * incoming_output
        return attn_mask, output

    # ## qatt7
    # def gratt_module(self, B, L, incoming_output, output, pred_prop_soft_list):  
    #     dec_out = self.decoder_norm(output)  # cQ, LB, E
    #     ## decide whether to propogate queries through frames
    #     # proj_output = self.prop_instance(dec_out)
    #     # print(f'[output] {output.shape} [dec_out] {dec_out.shape} [self.prop_instance(dec_out)]', proj_output.shape)
    #     pred_prop_hard, pred_prop_soft = self.gumble_gate(
    #         dec_out)  # [-1]  # take final layer output
    #     # it takes the gating output that attend the mask2former final layer
    #     # pred_prop_hard = pred_prop_hard.transpose(0, 1).view(L, B, self.num_queries, 1)[-1]  # .repeat(L, 1, 1).transpose(0,1)  # take the final hard pred
    #     # print('[pred_prop_soft]', pred_prop_soft.shape)
    #     # if self.training:
    #     #     pred_prop_soft_list.append(pred_prop_soft.transpose(0, 1).view(L, B, self.num_queries, 1))
    #     attn_mask = torch.zeros([B, self.num_queries, self.num_queries]).to(output)
    #     # mask_idx = pred_prop_hard[:, :, 0] == 0
    #     # attn_mask[mask_idx] = 1
    #     # for b in range(attn_mask.size(0)):
    #     #     attn_mask[b].fill_diagonal_(0)
    #     # attn_mask = attn_mask[None, :, None].repeat(L, 1, self.num_heads, 1, 1).view(L * B * self.num_heads, self.num_queries, self.num_queries)
    #     # pred_prop_hard = pred_prop_hard.repeat(L, 1, 1).transpose(0, 1)
    #     # print('repeat[pred_prop_hard]', pred_prop_hard.shape)
    #     output = pred_prop_soft * output + (1 - pred_prop_soft) * incoming_output
    #     return attn_mask, output

    def _window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBN, WTfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W

        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        # frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_query

    def _shift_window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(math.ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        # frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_query = torch.roll(frame_query, -half_W, 0)

        return frame_query
