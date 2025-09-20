# Meta Architecture for the U-Net with Mask2Former Decoder

import torch
import torch.nn as nn
import math
import os
import pkg_resources
from torch.nn import functional as F
from collections import defaultdict
from typing import List, Dict, Union
from ..mask2former import MultiScaleMaskedTransformerDecoder
# from .criterion import get_memm2f_criterion

## train_wrapper.py
from omegaconf import DictConfig, OmegaConf
import numpy as np
from einops.layers.torch import Rearrange

## sam2
from .build_sam2_modules import build_sam2_memory_encoder, build_sam2_memory_attention, build_sam2_pos_encoding
from ..SAM2.sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from torch.nn.init import trunc_normal_

def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

if is_installed("swattention"):
    from ..mask2former.transnext_unet_cuda import TransNeXtUNet
else:
    from ..mask2former.transnext_unet_native import TransNeXtUNet


track_train_cfg = OmegaConf.load("./model/memmask2former/train_config.yaml")
track_base_cfg = OmegaConf.load("./model/memmask2former/base.yaml")
track_model_cfg = OmegaConf.merge(track_train_cfg, track_base_cfg)

class MemMask2Former(nn.Module):
    def __init__(self, cfg, track_model_cfg):
        super().__init__()

        self.cfg, self.track_model_cfg = cfg, track_model_cfg
        self.freeze_backbone = cfg["Train"].get("freeze_backbone",False)
        # image size
        self.img_size = cfg["Model"]["image_size"]

        # Choose the encoder type of U-Net backbone
        encoder_type = cfg["Model"]["unet_backbone"]["encoder_type"]
        if encoder_type == "TransNeXt-Tiny":
            self.embed_dims = cfg["Model"]["unet_backbone"]["transnext_tiny_params"]["embed_dims"]
            self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["transnext_tiny_params"]["pretrained_path"]
        # elif encoder_type == "Swin-Small":
        #     self.embed_dims = cfg["Model"]["unet_backbone"]["swin_small_params"]["embed_dims"]
        #     self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["swin_small_params"]["pretrained_path"]
        elif encoder_type == "ConvNeXt":
            self.embed_dims = cfg["Model"]["unet_backbone"]["convnext_tiny_params"]["embed_dims"]
            self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["convnext_tiny_params"]["pretrained_path"]
        else:
            raise NotImplementedError(f"U-Net with {encoder_type} encoder is not implemented.")
        self.decoder_type = cfg["Model"]["unet_backbone"].get("decoder_type", "conv")
        
        # ------------------------------------
        # U-Net (backbone + decoder)
        # ------------------------------------
        if "TransNeXt" in encoder_type:
            if self.freeze_backbone:
                rank=4
            else:
                rank=0
            if track_model_cfg.shallow_backbone: #C2:
                self.embed_dims = self.embed_dims[:-1]
            self.unet_backbone = TransNeXtUNet(
                img_size=self.img_size,
                pretrain_size=self.img_size,
                embed_dims=self.embed_dims,
                decoder_type=self.decoder_type,
                num_heads=len(self.embed_dims),
                rank=rank
            )
        # elif "Swin" in encoder_type:
        #     if "tiny" in self.unet_backbone_pretrained_path:
        #         depths= [ 2, 2, 2, 2 ]
        #     elif "small" in self.unet_backbone_pretrained_path:
        #         depths=[2, 2, 18, 2]
        #     self.unet_backbone = SwinUNet(
        #         img_size=self.img_size,
        #         embed_dims=self.embed_dims,
        #         depths=depths,
        #         decoder_type=self.decoder_type
        #     )
        elif "ConvNeXt" in encoder_type:
            # if os.path.exists(self.unet_backbone_pretrained_path):
            print('[Convnext] from repo')
            from ..mask2former.convnext_repo_unet import ConvNextV2_UNet
            depths = [3, 3, 27, 3]
            if track_model_cfg.shallow_backbone: #C2:
                depths = [3, 3, 27]
                self.embed_dims = self.embed_dims[:-1]
            self.unet_backbone = ConvNextV2_UNet(depths=depths, img_size=self.img_size, 
                                                dims=self.embed_dims, 
                                                decoder_type=self.decoder_type)
            # else:
            #     from ..mask2former.convnext_unet import ConvNextV2_UNet
            #     pretrain_model = f"{self.unet_backbone_pretrained_path}-{self.img_size}"  # convnextv2-tiny-22k-224
            #     print(f"loading {pretrain_model}")
            #     self.unet_backbone = ConvNextV2_UNet.from_pretrained(f"{pretrain_model}",
            #                                                        img_size=self.img_size, 
            #                                                        embed_dims=self.embed_dims, 
            #                                                        decoder_type=self.decoder_type)
            #     print("finish loading")

        print('UNet params: ', sum(p.numel() for p in self.unet_backbone.parameters() if p.requires_grad))

        # ------------------------------------
        # Mask2Former Transformer Decoder
        # ------------------------------------
        if self.decoder_type == "pixel":  ## deform attn
            self.trans_decoder_in_channels, self.mask_dim = [256]*3, 256
        elif self.decoder_type == "pixelup":
            self.trans_decoder_in_channels, self.mask_dim = [256]*3, self.embed_dims[0]
        else:
            self.trans_decoder_in_channels = self.embed_dims[1:]
            self.mask_dim = self.embed_dims[0]
        
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels = self.trans_decoder_in_channels,
            hidden_dim = cfg["Model"]["mask2former_decoder"]["hidden_dim"],
            num_queries = cfg["Model"]["mask2former_decoder"]["num_queries"],
            nheads = cfg["Model"]["mask2former_decoder"]["nheads"],
            dim_feedforward = cfg["Model"]["mask2former_decoder"]["dim_feedforward"],
            dec_layers = cfg["Model"]["mask2former_decoder"]["dec_layers"],
            pre_norm = cfg["Model"]["mask2former_decoder"]["pre_norm"],
            mask_dim = self.mask_dim,
            mask_classification = cfg["Model"]["cls_head"],
            det_head = cfg["Model"]["det_head"],
            image_size = self.img_size,
            vis_query_att=cfg["Model"]["track_modules"].get("vis_query_att"),  ## 2/7 
            branch=cfg["Model"]["track_modules"].get("branch",1),  ## 3/24 
            short_term_t=cfg["Model"]["track_modules"].get("short_term_t",1),
            qatt_block=cfg["Model"]["track_modules"].get("qatt_block","crossatt"),
        )

        # ------------------------------------
        ## Memory modules
        # ------------------------------------
        # self.ms_dims = track_model_cfg.pixel_encoder.ms_dims
        # self.key_dim = track_model_cfg.key_dim
        self.value_dim = track_model_cfg.value_dim
        # self.sensory_dim = track_model_cfg.sensory_dim
        # self.pixel_dim = track_model_cfg.pixel_dim
        self.embed_dim = track_model_cfg.embed_dim
        self.use_amp = track_model_cfg.main_training.amp
        self.single_object=True

        ## Memory encoder
        if self.track_model_cfg.mask_encoder.method == "sam2":
            self.memory_encoder = build_sam2_memory_encoder(mem_dim=self.value_dim, image_size=self.img_size)
            # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
            self.sigmoid_scale_for_mem_enc = 20.0
            self.sigmoid_bias_for_mem_enc  = -10.0
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.value_dim)) # no_obj_embed_spatial: True
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        
        ## Memory attention
        self.object_transformer_enabled = track_model_cfg.object_transformer.enabled
        if self.track_model_cfg.memory_attention.method == "sam2":
            self.num_maskmem = 7  ## SAM2 default 1+6 (6 past frames to strike a balance between temporal context length and computational cost)
            self.memory_temporal_stride_for_eval = 1
            self.directly_add_no_mem_embed = True
            self.mem_dim, self.hidden_dim = self.value_dim, self.embed_dim
            local_branch = (self.track_model_cfg.memory_attention.branch == 2)
            spatial_local_CA = (local_branch and self.track_model_cfg.memory_attention.local_branch.spatial)
            if hasattr(self.memory_encoder, "out_proj") and hasattr(
                self.memory_encoder.out_proj, "weight"
            ):
                # if there is compression of memories along channel dim
                self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
            # Temporal encoding of the memories
            self.maskmem_tpos_enc = torch.nn.Parameter(
                torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
            )
            ## init embedding at t0
            self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.memory_attention = build_sam2_memory_attention(kv_in_dim=self.mem_dim,
                                                                local_branch=local_branch,
                                                                spatial_local_CA=spatial_local_CA)
            ## for ms_feat from image encoder (originaly in FPN neck)
            self.ms_feat_pos_enc = build_sam2_pos_encoding(num_pos_feats = 256, image_size=self.img_size)
            if self.track_model_cfg.mask_encoder.method != "sam2":
                ## for output_dict["maskmem_pos_enc"] (originaly in memory encoder)
                self.memory_pos_enc = build_sam2_pos_encoding(num_pos_feats=self.mem_dim, image_size=self.img_size)

            self.use_obj_ptrs_in_encoder = track_model_cfg.object_transformer.enabled
            self.add_tpos_enc_to_obj_ptrs = True
            self.proj_tpos_enc_in_obj_ptrs = True
            self.use_signed_tpos_enc_to_obj_ptrs = True
            self.only_obj_ptrs_in_the_past_for_eval = True
            self.pred_obj_scores = True ##
            self.pred_obj_scores_mlp = True ##
            self.fixed_no_obj_ptr = True ##
            self.use_mlp_for_obj_ptr_proj = True
            if self.use_obj_ptrs_in_encoder:
                # a linear projection on SAM output tokens to turn them into object pointers
                self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                if self.use_mlp_for_obj_ptr_proj:
                    self.obj_ptr_proj = MLP(
                        self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                    )
            else:
                self.obj_ptr_proj = torch.nn.Identity()
            if self.proj_tpos_enc_in_obj_ptrs and self.use_obj_ptrs_in_encoder:
                # a linear projection on temporal positional encoding in object pointers to
                # avoid potential interference with spatial positional encoding
                self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
            else:
                self.obj_ptr_tpos_proj = torch.nn.Identity()
        self.resample_memory = track_model_cfg.memory_bank.resample

    def load_pretrained_weight(self):
        # --------------------------------------------------------------------------
        ## Official Sam2 weight
        # --------------------------------------------------------------------------
        if self.track_model_cfg.mask_encoder.method == "sam2" or self.track_model_cfg.memory_attention.method == "sam2":
            if os.path.exists(self.track_model_cfg.sam2_weight_path):
                self.load_sam2_module_weight(self.track_model_cfg.sam2_weight_path)
                print(f"[LOAD] [SAM2 WEIGHT] Done.")
            else:
                print(f"[LOAD] [SAM2 WEIGHT] {self.track_model_cfg.sam2_weight_path} not found")

        # --------------------------------------------------------------------------
        # My pretrain weight
        # --------------------------------------------------------------------------
        ## m2f weight
        if os.path.exists(self.cfg["Model"]["mask2former_decoder"]["feat_extractor_ckpt_path"]):
            feat_extractor_checkpoint = torch.load(self.cfg["Model"]["mask2former_decoder"]["feat_extractor_ckpt_path"])
            if "n_averaged" in feat_extractor_checkpoint.keys():
                for key in list(feat_extractor_checkpoint.keys()):  ## remove "module." prefix if ema model is saved outside module
                    feat_extractor_checkpoint[key[7:]] = feat_extractor_checkpoint.pop(key)
            
            ## fix different input size
            if self.decoder_type == "pixel" and "pixup" in self.cfg["Model"]["mask2former_decoder"]["feat_extractor_ckpt_path"]:
                feat_extractor_checkpoint.pop('transformer_decoder.mask_embed.layers.2.weight')
                feat_extractor_checkpoint.pop('transformer_decoder.mask_embed.layers.2.bias')

            if self.track_model_cfg.shallow_backbone:
                for key in list(feat_extractor_checkpoint.keys()):
                    if 'pixel_decoder.input_convs' in key and 'conv.weight' in key:
                        new_dim = feat_extractor_checkpoint[key].shape[1] //2
                        feat_extractor_checkpoint[key] = feat_extractor_checkpoint[key][:,:new_dim,:,:]

            self.load_state_dict(feat_extractor_checkpoint, strict=False) 
            print("[LOAD] Feature Extractor checkpoint (M2F) weight Done.")

            ckpt_param = sum(p.numel() for p in feat_extractor_checkpoint.values())
            unet_param = sum(p.numel() for p in self.unet_backbone.parameters())
            trans_dec_param = sum(p.numel() for p in self.transformer_decoder.parameters())
            print(f'\n\tckpt params: {ckpt_param}\nunet {unet_param} + trans {trans_dec_param} = {unet_param+trans_dec_param}')        
            del not_in_ckpt, feat_extractor_checkpoint

        elif not os.path.exists(self.unet_backbone_pretrained_path):
            print(f"[LOAD] [BACKBONE] {self.unet_backbone_pretrained_path} not found.")
        else:
            checkpoint = torch.load(self.unet_backbone_pretrained_path)
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            if "state_dict" in checkpoint:  ## 512 pretrain weight ## rename the keys
                checkpoint = checkpoint["state_dict"]
                for key in list(checkpoint.keys()):
                    checkpoint[key.replace('backbone.', '')] = checkpoint.pop(key)
                for key in list(checkpoint.keys()):
                    checkpoint[key.replace('decode_head.', '')] = checkpoint.pop(key)
                
            self.unet_backbone.load_state_dict(checkpoint, strict=False)
            # self.transformer_decoder.load_state_dict(checkpoint, strict=False)

            print(f"[LOAD] [BACKBONE] Pretrained weight loaded!")
            del not_in_ckpt, checkpoint

    def rescale_pred_mask(self, out, images):
        ## https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py#L222
        ## if use pixel decoder, then predictions_cals[-1] has only 1/4 resolution
        if  out["pred_masks"].shape[-1] != images.shape[-1]:  # not self.training and
            out["origin_pred_masks"] = out["pred_masks"].clone()
            out["pred_masks"] = F.interpolate(
                out["pred_masks"],
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if self.training and out.get("aux_outputs"):  ## 
                for aux_output in out["aux_outputs"]:  # auxiliary masks
                    aux_output["origin_pred_masks"] = aux_output["pred_masks"].clone()
                    aux_output["pred_masks"] = F.interpolate(
                        aux_output["pred_masks"],
                        size=(images.shape[-2], images.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )

    ## Cutie functions ##

    def encode_memory(
            self,
            image: torch.Tensor,
            ms_features: Union[torch.Tensor, List[torch.Tensor]],  # original: one of the ms features (not sure why cutie wrote List[torch.Tensor] here)
            masks: torch.Tensor,
            *,
            object_score_logits = None) :  # -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        """
        The self.encode_mask function in Cutie
        - Input:
            - image
            - ms_features: 1st scale or [multi scale] feature from image encoder
            - masks: predicted mask
        - Output:
            - mask_value: feature of img+pred (concat or add) + ms_features (C1:list of small to large)
            - new_sensory (None)
            - object_summaries if self.object_transformer_enabled, else None. a list of feature if C1(3 scale)
            - object_logits (None)
            - maskmem_pos_enc
        """
        # image = (image - self.pixel_mean) / self.pixel_std
        # others = self._get_others(masks)
        if self.track_model_cfg.mask_encoder.method == "sam2":
            assert object_score_logits is not None
            new_sensory = None
            mask_value, maskmem_pos_enc = self._encode_new_memory(current_vision_feats = ms_features,  ## flattened input
                                                            pred_masks_high_res = masks,  ## binary
                                                            object_score_logits = object_score_logits)
            # print("[mask_value]", mask_value.shape)
            if isinstance(mask_value, list):
                object_summaries, object_logits = [None]*len(mask_value), [None]*len(mask_value)
            else:
                object_summaries, object_logits = None, None
        return mask_value, new_sensory, object_summaries, object_logits, maskmem_pos_enc

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    ## SAM2 functions ##

    def _prepare_memory_conditioned_features(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: list,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames=None,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        past_readouts = [], ## experiment
    ):
        """Fuse the current frame's visual feature map with previous memory.
        - current_vision_feats: list(1/16 flattened feature from image encoder w/ size [HW,B,C])
        - current_vision_pos_embeds:  list(pos encoded feature)
        - output_dict: cond_frame_outputs from add_new_points_or_box() in video predictor
        - num_frames: only matters when obj_ptr is used"""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes  # top-level (lowest-resolution) feature size (1/16)
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]  ## only t=0
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, -1 # self.max_cond_frames_in_attn  
            ) ## -1: not limited (but cond_frame is only t0 if naively operate)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            prev_t_list = list(output_dict["non_cond_frame_outputs"].keys())
            # print("prepare [unselected_cond_outputs]", unselected_cond_outputs.keys())
            # print("prepare [prev_t_list]", prev_t_list)
            temp = []
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    # else:
                    #     # the frame immediately after this frame (i.e. frame_idx + 1)
                    #     prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    # else:
                    #     # first find the nearest frame among every r-th frames after this frame
                    #     # for r=1, this would be (frame_idx + 2)
                    #     prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                    #     # then seek further among every r-th frames
                    #     prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                ## get all num_maskmem -1 features
                if len(prev_t_list) > 0:
                    prev_frame_idx = prev_t_list.pop(prev_t_list.index(max(prev_t_list)))
                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    temp.append(prev_frame_idx)
                    # print(f'prepare [t_pos] {t_pos} [prev_frame_idx from prev_t_list] {prev_frame_idx}')
                else:
                    out = None
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)  ## out=None if not enough cond_frame
                    # print(f'prepare [t_pos] {t_pos} [prev_frame_idx from unselected] {prev_frame_idx}')
                t_pos_and_prevs.append((t_pos, out))
            # if frame_idx < 30:
                # print(" [non_cond_frame_outputs] ", output_dict["non_cond_frame_outputs"].keys())
                # print(f"[non_cond_frame_outputs]: {temp}")
            ## NOTE!! t_pos small to large, prev feature from new to old
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))  ## flatten spatial dim
                
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"].to(device) ## maskmem_pos_enc is tensor, not list!
                # print("\n[maskmem_enc]", maskmem_enc.shape, "feats ",feats.shape)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]  ## [t_pos] #
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers  ## Similar ot E2? TODO
            if self.use_obj_ptrs_in_encoder:  
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:   ## True in SAM2.1
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim ## ptr_seq_len, B, C//256, 256 (neck dim)
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:  ## skip memory attention
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed: ## True in config base+ yaml
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)    ## [hwt, b, c]
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        if self.track_model_cfg.memory_attention.branch == 2:
            short_term_t = self.track_model_cfg.memory_attention.local_branch.short_term_t
            if len(to_cat_memory) > short_term_t:
                memory_local = torch.cat(to_cat_memory[:short_term_t], dim=0)    ## [hwt, b, c]
                memory_pos_embed_local = torch.cat(to_cat_memory_pos_embed[:short_term_t], dim=0)
            else:
                memory_local = torch.cat(to_cat_memory, dim=0)    ## [hwt, b, c]
                memory_pos_embed_local = torch.cat(to_cat_memory_pos_embed, dim=0)
        else:
            memory_local = None
            memory_pos_embed_local = None

        ## Experiment: Resampling the Memory Bank (Not used now)
        ## vlen = 8, training with this will make memory always len 7 (sampled by similarity)
        if self.resample_memory and len(past_readouts) > 0:
            memory_stack_ori = torch.stack(to_cat_memory,dim=0).permute(2,0,1,3) ## [b,t,hw,c] # memory.clone()
            memory_pos_stack_ori = torch.stack(to_cat_memory_pos_embed,dim=0).permute(2,0,1,3)#memory_pos_embed.clone()
            image_embed_stack_ori = torch.stack(past_readouts, dim=1)  ## to_cat_image_embed

            vision_feats_temp = current_vision_feats[-1].permute(1, 0, 2).reshape(B, -1) #, H, W 
            # print('[image_embed_stack_ori]', image_embed_stack_ori.shape, '[vision_feats_temp]',vision_feats_temp.shape)
            ## get similarity between current feature & past memory attention output
            image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=-1)
            vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=-1)           ## current feature
            
            similarity_scores = torch.bmm(image_embed_stack_ori, vision_feats_temp.unsqueeze(-1)).squeeze(-1)  # [b, t]
            # similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
            # print("[similarity_scores]", similarity_scores.shape)
            similarity_scores = F.softmax(similarity_scores, dim=1) 
            sampled_indices = torch.multinomial(similarity_scores, num_samples=self.num_maskmem, replacement=True).squeeze(1)  # souce code: Shape [B, 16](not sure why 16)
            # print("[sampled_indices]", sampled_indices, '[memory_stack_ori]',memory_stack_ori.shape)
            expanded_indices = sampled_indices[:, :, None, None].expand(-1, -1, memory_stack_ori.shape[-2], memory_stack_ori.shape[-1])  # Shape [b, 7, hw, c]
            memory_stack_ori_new = torch.gather(memory_stack_ori, dim=1, index=expanded_indices).flatten(1,2)
            # print('[memory_stack_ori_new]', memory_stack_ori_new.shape)
            memory = memory_stack_ori_new.permute(1,0,2)
            # print("\t[memory (new)]",memory.shape)
            memory_pos_stack_new = torch.gather(memory_pos_stack_ori, dim=1, index=expanded_indices).flatten(1,2)
            memory_pos_embed = memory_pos_stack_new.permute(1,0,2)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            memory_local=memory_local,
            memory_pos_local=memory_pos_embed_local,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)  ## mem_dim, H/16, W/16
        # print("[pix_feat_with_mem]", pix_feat_with_mem.shape)
        return pix_feat_with_mem, torch.permute(memory, (1, 0, 2))
    
    def _origin_prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        return
        """Fuse the current frame's visual feature map with previous memory.
        - current_vision_feats: list of 1/4~1/16 features 
                                (Image encoder discard the lowest resolution features, 
                                so backbone_fpn contains only 1/4~1/16)"""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]  ## only t=0
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, -1 # self.max_cond_frames_in_attn  
            ) ## -1 for (a): only allow N prev feature maps
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)  ## out=None if not enough cond_frame
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))  ## flatten spatial dim
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)  ## get the last one from list
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:   ## True in SAM2.1
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim ## ptr_seq_len, B, C//256, 256 (neck dim)
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:  ## skip memory attention
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed: ## True in config base+ yaml
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def get_curr_vision_feat_pos(self, multi_scale_features, mask_feature):
        """
        Postion encoding on image encoder output feature & flatten feature \\
        Return 
            - current_vision_feats & current_vision_pos_embeds (large to small) w/ size [HW, B, C]
            - pix_feat (small to large) w/ size [B, C, H, W]
        """
        # pix_feat = multi_scale_features  ## smallest 3 feature
        pix_feat = multi_scale_features + [mask_feature]  ## all 4 features

        ## SAM2 FPN applies pos encoding on ms feature 
        current_vision_pos_embeds = [self.ms_feat_pos_enc(feat) for feat in pix_feat]
        ## flatten
        current_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in pix_feat]
        current_vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in current_vision_pos_embeds]
        current_vision_feats.reverse() ## large to small
        current_vision_pos_embeds.reverse()
        # for curr_feat in current_vision_feats:
        #     print('[curr_feat]', curr_feat.shape)
        if self.track_model_cfg.mem_scale == 1 and not self.track_model_cfg.shallow_backbone: #A2 and not self.track_model_cfg.C2: ## return largest 3 features 1/4~1/16
            current_vision_feats = current_vision_feats[:-1]  ## remove the smallest
            current_vision_pos_embeds = current_vision_pos_embeds[:-1]
            pix_feat = pix_feat[1:]


        return current_vision_feats, current_vision_pos_embeds, pix_feat
    
    def _encode_new_memory(
        self,
        current_vision_feats, ## large to small (1/4, 1/4, 1/8, 1/16)
        pred_masks_high_res,
        object_score_logits,
        # is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature. \\
        Return maskmem_features & maskmem_pos_enc"""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        # if self.non_overlap_masks_for_mem_enc and not self.training:  ## False
        #     # optionally, apply non-overlapping constraints to the masks (it's applied
        #     # in the batch dimension and should only be used during eval, where all
        #     # the objects come from the same video under batch size 1).
        #     pred_masks_high_res = self._apply_non_overlapping_constraints(
        #         pred_masks_high_res
        #     )
        # scale the raw mask logits with a temperature before applying sigmoid
        # binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts  ## False
        if False and not self.training: # and binarize
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        
        if self.track_model_cfg.mem_scale > 1:#.C1:
            ms_maskmem_features, ms_maskmem_pos_enc = [],[]
            scale_list = [32,16,8]
            if self.track_model_cfg.shallow_backbone: #C2:
                scale_list = [16,8,4]
            for s in range(self.track_model_cfg.mem_scale):
                H, W = self.img_size//scale_list[s],self.img_size//scale_list[s] ## small to large
                # (HW)BC => BCHW
                current_vision_feats[3-s] = current_vision_feats[3-s].permute(1, 2, 0).view(B, C, H, W)
            current_vision_feats = current_vision_feats[-self.track_model_cfg.mem_scale:]  ## only keep the scales needed
            maskmem_out = self.memory_encoder(
                current_vision_feats, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
            )
            for s in range(self.track_model_cfg.mem_scale):
                maskmem_features = maskmem_out["vision_features"][s]  ## [B, value_dim, H//64, W//64]
                maskmem_pos_enc = maskmem_out["vision_pos_enc"][s]
                if isinstance(maskmem_pos_enc,list):  ## memory_encoder output list(maskmem_pos_enc)
                    maskmem_pos_enc = maskmem_pos_enc[-1]
                # add a no-object embedding to the spatial memory to indicate that the frame
                # is predicted to be occluded (i.e. no object is appearing in the frame)
                if object_score_logits.dim() > 2:
                    object_score_logits = object_score_logits.squeeze(-1)  ## -> [B,1]

                if self.no_obj_embed_spatial is not None:
                    is_obj_appearing = (object_score_logits > 0).float()
                    maskmem_features += (
                        1 - is_obj_appearing[..., None, None]
                    ) * self.no_obj_embed_spatial[..., None, None].expand(
                        *maskmem_features.shape
                    )
                ms_maskmem_features.append(maskmem_features)
                ms_maskmem_pos_enc.append(maskmem_pos_enc)
            return ms_maskmem_features, ms_maskmem_pos_enc

        elif self.track_model_cfg.mem_scale == 1: #A2:  ## TODO (low prio) modulize
            H, W = self.img_size//16,self.img_size//16 # SAM2: top-level (lowest-resolution) feature size
  
            # top-level feature, (HW)BC => BCHW
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            maskmem_out = self.memory_encoder(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
            )
            maskmem_features = maskmem_out["vision_features"]  ## [B, value_dim, H//64, W//64]
            maskmem_pos_enc = maskmem_out["vision_pos_enc"]
            if isinstance(maskmem_pos_enc,list):  ## memory_encoder output list(maskmem_pos_enc)
                maskmem_pos_enc = maskmem_pos_enc[-1]
            # add a no-object embedding to the spatial memory to indicate that the frame
            # is predicted to be occluded (i.e. no object is appearing in the frame)
            if object_score_logits.dim() > 2:
                object_score_logits = object_score_logits.squeeze(-1)  ## -> [B,1]

            if self.no_obj_embed_spatial is not None:
                is_obj_appearing = (object_score_logits > 0).float()
                maskmem_features += (
                    1 - is_obj_appearing[..., None, None]
                ) * self.no_obj_embed_spatial[..., None, None].expand(
                    *maskmem_features.shape
                )

        return maskmem_features, maskmem_pos_enc
    
    def load_sam2_module_weight(self, ckpt_path) -> None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        for k in list(sd.keys()):
            if "image_encoder" in k or "sam_prompt_encoder" in k or "sam_mask_decoder" in k:
                sd.pop(k)
        if self.memory_attention.num_layers > 4:   ## not used now
            new_checkpoint = sd.copy()
            for name, param in sd.items():
                if name.startswith("memory_attention.layers.2"):
                    new_name = name.replace("memory_attention.layers.2", "memory_attention.layers.4")
                    new_checkpoint[new_name] = param.clone()
                elif name.startswith("memory_attention.layers.3"):
                    new_name = name.replace("memory_attention.layers.3", "memory_attention.layers.5")
                    new_checkpoint[new_name] = param.clone()
            sd = new_checkpoint
        if self.cfg["Model"]["track_modules"].get("query_membank_att"):   ## not used now
            new_checkpoint = sd.copy()
            for name, param in sd.items():
                if name.startswith("memory_attention.layers.0.cross_attn_image"):
                    new_name = name.replace("memory_attention.layers.0.cross_attn_image", "transformer_decoder.query_membank_att_layers.0")
                    new_checkpoint[new_name] = param.clone()
                elif name.startswith("memory_attention.layers.1.cross_attn_image"):
                    new_name = name.replace("memory_attention.layers.1.cross_attn_image", "transformer_decoder.query_membank_att_layers.1")
                    new_checkpoint[new_name] = param.clone()
                elif name.startswith("memory_attention.layers.2.cross_attn_image"):
                    new_name = name.replace("memory_attention.layers.2.cross_attn_image", "transformer_decoder.query_membank_att_layers.2")
                    new_checkpoint[new_name] = param.clone()
            sd = new_checkpoint
        if self.track_model_cfg.memory_attention.branch == 2:
            new_checkpoint = sd.copy()
            for name, param in sd.items():
                if name.startswith("memory_attention.layers.0.cross_attn_image"):
                    new_name = name.replace("memory_attention.layers.0.cross_attn_image", "memory_attention.layers.0.cross_attn_image_local")
                    new_checkpoint[new_name] = param.clone()
                elif name.startswith("memory_attention.layers.1.cross_attn_image"):
                    new_name = name.replace("memory_attention.layers.1.cross_attn_image", "memory_attention.layers.1.cross_attn_image_local")
                    new_checkpoint[new_name] = param.clone()
                # elif name.startswith("memory_attention.layers.2.cross_attn_image"):
                #     new_name = name.replace("memory_attention.layers.2.cross_attn_image", "memory_attention.layers.2.cross_attn_image_local")
                #     new_checkpoint[new_name] = param.clone()
            sd = new_checkpoint
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        if unexpected_keys:
            not_in_model = set()
            for name in unexpected_keys:
                if 'cross_attn_image'  in name:
                    not_in_model.add(name[:47])
            print("Unexpected keys (sam2 parameters not found in model):")
            for name in not_in_model:
                print('\t',name)
        del sd, missing_keys, unexpected_keys


class MemTrainWrapper(MemMask2Former):
    def __init__(self, cfg, track_model_cfg):
        super().__init__(cfg, track_model_cfg)
        # single_object=(track_model_cfg.main_training.num_objects == 1)
        # self.sensory_dim = track_model_cfg.sensory_dim
        self.seq_length = track_model_cfg.main_training.seq_length
        self.use_amp = track_model_cfg.main_training.amp
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=self.seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=self.seq_length)
        
        self.criterion = get_memm2f_criterion(cfg, track_model_cfg)
        self.device = "cuda" # "cpu"#
        self.LCR = self.cfg["Data"].get("video_LCR", False)

        self.tbtt_k1, self.tbtt_k2 = -1, -1

    def get_ms_feat_ti(self, ti, ms_feat):
        return [f[:, ti] for f in ms_feat]    
    
    ## trainwrapper.forward + trainer.do_pass + sam2_base function
    def do_pass_sam2_memory_attention(self, batched_inputs, calculate_loss=True, synthetic_pret=False, t0_gt=False, optimizer=None):
        
        ## final output
        seq_pred_masks, seq_pred_class = [],[]
        
        ## init memory for vis_query_att
        if (self.cfg["Model"]["track_modules"].get("vis_query_att") or 
            (self.cfg["Model"]["track_modules"]["moe_q"] is not None and 'past' in self.cfg["Model"]["track_modules"]["moe_q"])):
            pre_memory = {"k": [], "v": []}
        elif self.cfg["Model"]["track_modules"].get("query_membank_att"):
            pre_memory = {}
        else:
            pre_memory = None
        
        ## resample memory (the lists will be called if self.resample_memory)
        if self.track_model_cfg.mem_scale == 3:#.C1:
            past_readouts_list = [[],[],[]]
        elif self.track_model_cfg.mem_scale == 2:
            past_readouts_list = [[],[]]
        elif self.track_model_cfg.mem_scale == 1:#A2:
            past_readouts_list = []
        
        ## init loss
        if "tc" in self.cfg["Train"]["loss_weight"]:
            prev_pred_masks = []
        losses = defaultdict(float)  ## default 0.0

        ## output_dict: Record memory of SAM2
        ## add t as key, {"maskmem_features": encoded memory, "maskmem_pos_enc":pos embedding} as value
        if self.track_model_cfg.mem_scale > 1: #.C1:
            output_dict = [{"cond_frame_outputs":{}, "non_cond_frame_outputs":{}}]*self.track_model_cfg.mem_scale
        else:    
            output_dict = {"cond_frame_outputs":{}, "non_cond_frame_outputs":{}}

        ## get GT
        if synthetic_pret:
            frame_masks = batched_inputs["masks"]         ## [B,T,H,W]
            frame_cals = batched_inputs.get("cals",None)           ## [B,T,4]
            frame_endpoints = batched_inputs.get("endpoints",None) ## [B,T,4]
            frame_labels = batched_inputs.get("labels",None)       ## [B,T]
        elif self.LCR:
            frame_masks = batched_inputs["masks"][:,1:-1,:,:]       ## [B,T,H,W]  ## remove first time_window 
            frame_cals = batched_inputs["cals"][:,1:-1,:]           ## [B,T,4]
            frame_endpoints = batched_inputs["endpoints"][:,1:-1,:] ## [B,T,4]
            frame_labels = batched_inputs["labels"][:,1:-1]         ## [B,T]
        else:
            frame_masks = batched_inputs["masks"][:,2:,:,:]       ## [B,T,H,W]  ## remove first time_window 
            frame_cals = batched_inputs["cals"][:,2:,:]           ## [B,T,4]
            frame_endpoints = batched_inputs["endpoints"][:,2:,:] ## [B,T,4]
            frame_labels = batched_inputs["labels"][:,2:]         ## [B,T]
        batch_size = frame_masks.shape[0]
        video_len = frame_masks.shape[1]
        ## NOTE: input frame one by one to avoid CUDA memory error
        
        ## TBTT (Not used now)
        # self.set_tbtt(frame_masks)
        if self.tbtt_k2 > 0:
            detach_output_dict = {"cond_frame_outputs":{}, "non_cond_frame_outputs":{}} 
        
        for ti in range(video_len):
            # print(f"\n=== {ti} ===")
            # --------------------------------------------------------------------------
            # Get all images
            # --------------------------------------------------------------------------
            if synthetic_pret:  ## (Not used now)
                image = batched_inputs["images"][:, ti, :,:]
                image = image.unsqueeze(1).expand(-1,3,-1,-1) ## channel 1->3
            else:    
                image = batched_inputs["images"][:, ti: ti+3, :,:]
            image = image.to(self.device)
            # ## expiriment A1: drop path rate for ti!=0
            # if ti == 0:
            #     self.unet_backbone.set_drop_path_rate(0.0)
            # else:
            #     self.unet_backbone.set_drop_path_rate(0.6)
            ## expiriment: TBTT (Not used now)
            n_c_key = "non_cond_frame_outputs"
            if self.tbtt_k2 > 0 and len(output_dict["non_cond_frame_outputs"]) > self.tbtt_k2:
                ## Delete stuff that is too old
                # del output_dict[n_c_key][min(output_dict[n_c_key].keys())]
            # if self.tbtt_k1 > 0 and len(output_dict["non_cond_frame_outputs"]) > 0:
                # detach_key = output_dict[n_c_key].keys()[-self.tbtt_k2 - 1] ## every feature detach
                # output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_features"] = \
                #     output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_features"].detach()
                # output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_features"].requires_grad=True
                # output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_pos_enc"] = \
                #     output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_pos_enc"].detach()
                # output_dict[n_c_key][max(output_dict[n_c_key].keys())]["maskmem_pos_enc"].requires_grad=True
                detach_key = list(output_dict[n_c_key].keys())[-self.tbtt_k2 - 1] ## only the last k2 features keep grad
                # detach_key = max(output_dict[n_c_key].keys()) ## keep no grad
                detach_output_dict[n_c_key][detach_key] = {}
                detach_output_dict[n_c_key][detach_key]["maskmem_features"] = \
                    output_dict[n_c_key][detach_key]["maskmem_features"].detach()
                detach_output_dict[n_c_key][detach_key]["maskmem_features"].requires_grad=True
                detach_output_dict[n_c_key][detach_key]["maskmem_pos_enc"] = \
                    output_dict[n_c_key][detach_key]["maskmem_pos_enc"].detach()
                detach_output_dict[n_c_key][detach_key]["maskmem_pos_enc"].requires_grad=True
            if self.tbtt_k1 > 0 and len(output_dict["cond_frame_outputs"]) > 0 and ti < 2: 
                detach_output_dict["cond_frame_outputs"][0] = {}
                detach_output_dict["cond_frame_outputs"][0]["maskmem_features"] = \
                    output_dict["cond_frame_outputs"][0]["maskmem_features"].detach()
                detach_output_dict["cond_frame_outputs"][0]["maskmem_features"].requires_grad=True
                detach_output_dict["cond_frame_outputs"][0]["maskmem_pos_enc"] = \
                    output_dict["cond_frame_outputs"][0]["maskmem_pos_enc"].detach()
                detach_output_dict["cond_frame_outputs"][0]["maskmem_pos_enc"].requires_grad=True

            # --------------------------------------------------------------------------
            # M2f: backbone encoder + pixel decoder
            # --------------------------------------------------------------------------
            # multi_scale_features, mask_feature = self.unet_backbone(image)  ## mask_feature [B, C=72, H, W]  ## multi_scale_features [B, C', H', W']*3 (1/32, 1/16, 1/8)
            multi_scale_features = self.unet_backbone.forward_features(image)  ## encoder
            if self.track_model_cfg.Neck == 2:
                multi_scale_features, mask_feature = self.unet_backbone.upward_features(multi_scale_features) ## pixel decoder
            else:
                multi_scale_features.reverse()   ## ->  small to big
            # print(f"mask_feature {mask_feature.shape} multi_scale_features {multi_scale_features[0].shape}{multi_scale_features[1].shape}{multi_scale_features[2].shape}")
            # assert torch.isnan(multi_scale_features[0]).any() == False
            # assert torch.isinf(multi_scale_features[0]).any() == False

            # --------------------------------------------------------------
            # Get pix_feat: project multi scale feature from encoder
            # & position embedding as SAM2 FPN
            # --------------------------------------------------------------
            (current_vision_feats, 
            current_vision_pos_embeds,
            pix_feat) = self.get_curr_vision_feat_pos(multi_scale_features, mask_feature)  ## pix_feat only used in cutie encoder
            
            if ti == 0:
                # --------------------------------------------------------------
                # Init prediction as GT mask at first frame 
                # --------------------------------------------------------------
                if t0_gt or synthetic_pret:
                    first_frame_gt = frame_masks[:,0,:,:].unsqueeze(1).to(self.device)
                else:
                    if self.track_model_cfg.Neck == 1:
                        multi_scale_features.reverse() ## -> big to small, 4 levels
                        multi_scale_features, mask_feature = self.unet_backbone.upward_features(multi_scale_features) ## pixel decoder
                    out, _ = self.transformer_decoder(multi_scale_features, mask_feature)
                    self.rescale_pred_mask(out, image)
                    first_frame_gt = (out["pred_masks"] > 0.5).float()  ## [B,1,H,W]
                masks = first_frame_gt  ## [B,1,H,W]  ## assume this is GT
                
                if self.resample_memory:  ## (Not used now)
                    if self.track_model_cfg.mem_scale > 1:#.C1:
                        for s in range(self.track_model_cfg.mem_scale):
                            # print(f"[current_vision_feats[{3-s}] t0]", current_vision_feats[3-s].permute(1, 0, 2).reshape(batch_size,-1).shape)
                            past_readouts_list[s].append(current_vision_feats[self.track_model_cfg.mem_scale-s].permute(1, 0, 2).reshape(batch_size,-1).detach())  ## small to large
                    elif self.track_model_cfg.mem_scale == 1:#.A2:
                        past_readouts_list[s].append(current_vision_feats[-1].permute(1, 0, 2).reshape(batch_size,-1).detach())
                # --------------------------------------------------------------
                # Encode mask memory: msk_val
                # Deep update sensory
                # --------------------------------------------------------------
                with torch.cuda.amp.autocast(enabled=False):  # self.use_amp
                    # found_seq_first_pred = (out["pred_masks"] > 0.5).all(dim=[1, 2, 3], keepdim=True)  ## if each sequence's first needle is found or not (not used now)
                    if self.track_model_cfg.mask_encoder.method == "sam2":
                        msk_val, _, obj_val, _, maskmem_pos_enc = self.encode_memory(image, current_vision_feats,
                                                                    masks=masks,
                                                                    object_score_logits = out.get("pred_class", None))
                    if self.track_model_cfg.mask_encoder.method != "sam2": ## TODO (low prio) C1
                        maskmem_pos_enc = self.memory_pos_enc(msk_val)
                    # NOTE A2 msk_val [B, 256, 1, H/16, W/16]; C1 1/32~1/8
                    
                # --------------------------------------------------------------

                # --------------------------------------------------------------
                # Accumulate memory features
                ## Naively view t0 as conditioned
                # --------------------------------------------------------------
                if isinstance(msk_val, list):  ## if self.track_model_cfg.C1:
                    # print(f"[msk_val] list {msk_val[0].shape} {msk_val[1].shape}")
                    for s in range(len(msk_val)):
                        output_dict[s]["cond_frame_outputs"][0] = {"maskmem_features": msk_val[s].squeeze(1),  # B,C,H',W'
                                                                "maskmem_pos_enc": maskmem_pos_enc[s]}
                else:
                    output_dict["cond_frame_outputs"][0] = {"maskmem_features": msk_val.squeeze(1),  # B,C,H',W' ## NOTE only 1/16!! for A2
                                                        "maskmem_pos_enc": maskmem_pos_enc} 
                
                # print("[msk_val] ", msk_val.shape)
                # print("[maskmem_pos_enc] ", maskmem_pos_enc.shape)
                # obj_values = obj_val  ##  (time dim added in encode_memory)  B*num_objects*t*Q*C
                if self.tbtt_k2 > 0:
                    detach_output_dict["cond_frame_outputs"][0] = {"maskmem_features": msk_val.squeeze(1),  # B,C,H',W'
                                                        "maskmem_pos_enc": maskmem_pos_enc} 
                if t0_gt or synthetic_pret:
                    continue ## no loss for first frame

            # if not found_seq_first_pred.all():
            #     new_found_pred = (out["pred_masks"] > 0.5).all(dim=[1, 2, 3], keepdim=True)
            #     find_first_now = ~found_seq_first_pred & new_found_pred  ## needle not found in previous t but found now
            #     found_seq_first_pred = new_found_pred | found_seq_first_pred  ## update: found now or previously found
                
            #     # Replace the entire tensor first_frame_gt[i] with pred_masks[i] if find first
            #     first_frame_gt = torch.where(find_first_now, (out["pred_masks"] > 0.5).float(), first_frame_gt)

            # --------------------------------------------------------------------------
            # Segment frame ti
            # --------------------------------------------------------------------------
            ## check if needle in ouptut, if not then not do memory attention on next frame? TODO
            ## so should aux pred TODO
            ## No need to do memory attention on the first frame since memory bank only contains itself
            obj_aux_output = None
            if ti > 0:
                # --------------------------------------------------------------
                # Get "Readout": Read from memories to get a memory affected feature (default 1/16 for sam2)
                # --------------------------------------------------------------
                if self.track_model_cfg.mem_scale > 1:#.C1:
                    readout, scale_list = [],[32,16,8]
                    if self.track_model_cfg.shallow_backbone: #C2:
                        scale_list = [16,8,4]
                    for s in range(self.track_model_cfg.mem_scale):
                        readout_s, bank_memory = self._prepare_memory_conditioned_features(
                                        frame_idx = ti,
                                        is_init_cond_frame = False,  ## SAM2 only treat frames with prompt as cond_frame
                                        current_vision_feats = [current_vision_feats[3-s]],  
                                        current_vision_pos_embeds = [current_vision_pos_embeds[3-s]],
                                        feat_sizes = (self.img_size//scale_list[s], self.img_size//scale_list[s]),
                                        output_dict = output_dict[s],
                                        num_frames=video_len,
                                        past_readouts=past_readouts_list[s])
                        readout.append(readout_s.squeeze(1)) ## smallest (1/32) to largest(1/8) scale [2, 256, 24, 24]
                        if self.resample_memory:
                            # print("[readout_s]", readout_s.shape)
                            past_readouts_list[s].append(readout_s.reshape(batch_size,-1).detach())
                else:
                    readout, bank_memory = self._prepare_memory_conditioned_features(
                        frame_idx = ti,
                        is_init_cond_frame = False,  ## SAM2 only treat frames with prompt as cond_frame
                        current_vision_feats = current_vision_feats[-1:],  ## list of smallest feature in 3 largest scale
                        current_vision_pos_embeds = current_vision_pos_embeds[-1:],
                        feat_sizes = (self.img_size//16, self.img_size//16),
                        output_dict = output_dict, ## output_dict,  ## 
                        num_frames=video_len,
                        past_readouts=past_readouts_list)
                    if self.resample_memory:
                        past_readouts_list.append(readout.reshape(batch_size,-1).detach())
                
                if self.cfg["Model"]["track_modules"].get("query_membank_att"):
                    pre_memory["bank_memory"] = bank_memory
                # --------------------------------------------------------------
                # Decode with multi scale features and "Readout"
                # Feature from image encoder is replaced with Readout
                # --------------------------------------------------------------
                ## Mask decoder in SAM2 (self.segment) replace with transformer_decoder
                ## out: dict("pred_masks","pred_class", "aux_outputs")
                if self.track_model_cfg.mem_scale == 1: #not self.track_model_cfg.C1:
                    decoder_input_readout = readout.clone().squeeze(1)
                if self.track_model_cfg.mem_scale > 1: #.C1:
                    if self.track_model_cfg.ms_readout is not None and "norm" in self.track_model_cfg.ms_readout:
                        readout = [self.read_norm[i](readout[i]) for i in range(len(readout))]
                        multi_scale_features = [self.feat_norm[i](multi_scale_features[i]) for i in range(len(readout))]
                    if self.track_model_cfg.ms_readout is not None and "sum" in self.track_model_cfg.ms_readout:
                        readout = [mem_read + ms_feat for mem_read, ms_feat in zip(readout, multi_scale_features)]
                        if self.track_model_cfg.ms_readout is not None and "relu" in self.track_model_cfg.ms_readout:
                            readout = [self.activ(mem_read) for mem_read in readout]
                    if self.track_model_cfg.Neck == 1:
                        readout.reverse() ## -> big to small
                        # print("upward", multi_scale_features[-1].shape)
                        # for ro in readout:
                        #     print(ro.shape)
                        readout, mask_feature = self.unet_backbone.upward_features([multi_scale_features[-1]] + readout) ## pixel decoder
                    if len(readout) == 2:  ## mem_scale: 2
                        readout = readout + [multi_scale_features[2]]
                        # for re in readout:
                        #     print(f'[ms=2]{re.shape}')
                    out, _ = self.transformer_decoder(readout, mask_feature, pre_memory=pre_memory) #### readout is a list of augmented ms_feature (small to big)
                elif self.track_model_cfg.mem_scale == 1: #.A2:
                    if self.track_model_cfg.ms_readout == "avg":
                        decoder_input_readout = (readout.squeeze(1) + multi_scale_features[1]) /2
                        sensory_input_readout = (readout + multi_scale_features[1].unsqueeze(1)) /2
                    if self.track_model_cfg.B1 :
                        ## MaskUpsampleBlock only needs readout with num_obj dim, ouput is also with num_obj dim
                        multi_scale_features[2] = self.up_16_8(decoder_input_readout.unsqueeze(1), multi_scale_features[2]).squeeze(1)
                    # print("Replace 1/16 feature for decoder")
                    if self.track_model_cfg.shallow_backbone: #C2:
                        readout = [decoder_input_readout, multi_scale_features[1], multi_scale_features[2]]
                    else:
                        readout = [multi_scale_features[0], decoder_input_readout, multi_scale_features[2]]
                    if self.track_model_cfg.Neck == 1:
                        readout.reverse() ## -> big to small  ## TODO?
                        readout, mask_feature = self.unet_backbone.upward_features(readout) ## pixel decoder
                    out, _ = self.transformer_decoder(readout, mask_feature, pre_memory=pre_memory) #### A2
                
                assert torch.isnan(out["pred_masks"]).any() == False
                assert torch.isinf(out["pred_masks"]).any() == False 
                if (self.cfg["Model"]["track_modules"].get("vis_query_att") or 
                    (self.cfg["Model"]["track_modules"]["moe_q"] is not None and 'past' in self.cfg["Model"]["track_modules"]["moe_q"])):
                    pre_memory["k"].append(out["pre_memory"]["k"])
                    pre_memory["v"].append(out["pre_memory"]["v"])
                # --------------------------------------------------------------
                # Reset the latest pred mask
                # --------------------------------------------------------------
                self.rescale_pred_mask(out, image)
                masks = (out["pred_masks"] > 0.5).float()  ## TODO cls head zero_?
                # --------------------------------------------------------------

                
                # --------------------------------------------------------------
                # Encode new memory
                # No need to encode the last frame
                # --------------------------------------------------------------
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if ti < (self.seq_length - 1):
                        if self.track_model_cfg.mask_encoder.method == "sam2":
                            msk_val, _, obj_val, _, maskmem_pos_enc = self.encode_memory(image, current_vision_feats, 
                                                                        masks=masks, 
                                                                        object_score_logits = out.get("pred_class", None))
                        # if self.track_model_cfg.mask_encoder.method != "sam2":
                        #     maskmem_pos_enc = self.memory_pos_enc(msk_val)

                        # print("[maskmem_pos_enc] ", maskmem_pos_enc.shape)
                        # obj_values = obj_val  ##  (time dim added in encode_memory)  B*num_objects*t*Q*C
                        if isinstance(msk_val, list):
                            for s in range(len(msk_val)):
                                output_dict[s]["non_cond_frame_outputs"][ti] = {"maskmem_features": msk_val[s].squeeze(1),
                                                                                "maskmem_pos_enc": maskmem_pos_enc[s]}
                        else:
                            output_dict["non_cond_frame_outputs"][ti] = {"maskmem_features": msk_val.squeeze(1), ## remove obj dim 
                                                                        "maskmem_pos_enc": maskmem_pos_enc} 
                        if self.tbtt_k2 > 0:
                            detach_output_dict["non_cond_frame_outputs"][ti] = {"maskmem_features": msk_val.squeeze(1),  ## remove obj dim 
                                                                    "maskmem_pos_enc":maskmem_pos_enc}

            del multi_scale_features, mask_feature
            
            # --------------------------------------------------------------------------
            # Get Loss (ema model does not need loss)
            # --------------------------------------------------------------------------
            if self.criterion is not None and calculate_loss:  
                ## Get Loss
                if obj_aux_output is not None:
                    out = {**out, **obj_aux_output}
                if "tc" in self.cfg["Train"]["loss_weight"]:
                    prev_pred_masks.append(out["pred_masks"])
                    out["prev_pred_masks"] = prev_pred_masks
                    out["batch_images"] = batched_inputs["images"][:,: ti+3, :,:]
                if isinstance(frame_labels, torch.Tensor) and isinstance(frame_endpoints, torch.Tensor):
                    step_loss_dict = self.criterion(out, frame_masks[:,ti], frame_cals[:,ti], frame_endpoints[:,ti], frame_labels[:,ti])
                else:
                    step_loss_dict = self.criterion(out, frame_masks[:,ti], frame_cals, frame_endpoints, frame_labels)
                # print("step_loss_dict ",step_loss_dict)
                for k in list(step_loss_dict.keys()):
                    if '-' in k:  ## aux layer
                        layer_idx = k.index('-')
                        k_l = k[:layer_idx]
                    else:
                        k_l = k
                    losses[k_l] += step_loss_dict[k]#.item() #* genvis_weight_dict[k]
           
                if out.get("moe_loss"):
                    losses['moe_loss'] += out.get("moe_loss")
                # print(f"losses t{ti}",losses)
                
                # ## accumulate gradient
                # ## https://stackoverflow.com/questions/63934070/gradient-accumulation-in-an-rnn
                # loss = sum(v for k in step_loss_dict.values()) 
                # loss = loss / video_len 
                # loss.backward()
                # if self.use_amp:
                #     self.scaler.scale(loss).backward()
                
                ## https://stackoverflow.com/questions/62901561/truncated-backpropagation-in-pytorch-code-check
                if self.tbtt_k1 > 0 and (ti+1) % self.tbtt_k1 == 0:
                    # print("[backward]")
                    loss_backward = sum(v for v in losses.values()) 
                    loss_backward.backward()
                    assert optimizer is not None
                    optimizer.step()
                    optimizer.zero_grad()
                    if ti != video_len-1:  ## not the last t
                        del losses
                        losses = defaultdict(float)

            seq_pred_masks.append(out["pred_masks"])
            seq_pred_class.append(out["pred_class"])
        
            ## end of a time step ##

        seq_pred_masks = torch.cat(seq_pred_masks, dim=1)
        seq_pred_masks = seq_pred_masks.detach()
        if isinstance(seq_pred_class[0], torch.Tensor):
            seq_pred_class = torch.cat(seq_pred_class, dim=1)
            seq_pred_class = seq_pred_class.detach()
        batched_inputs["images"] = batched_inputs["images"].detach().cpu()
        batched_inputs["masks"] = batched_inputs["masks"].detach().cpu()

        pred_ouptuts = {
            "pred_masks": seq_pred_masks,
            "pred_class": seq_pred_class,
            "visualize_images": batched_inputs["images"][-1][2:,:,:], ## [T,H,W]
            "visualize_masks":  batched_inputs["masks"][-1][2:,:,:]
        }
        if "experts_weight" in out:
            pred_ouptuts["experts_weight"] = out.get('experts_weight')[0].flatten()

        del batched_inputs
        return losses, pred_ouptuts


    def forward(self, batched_inputs, calculate_loss=True, t0_gt=False, optimizer=None):
        if self.track_model_cfg.memory_attention.method == "sam2":
            do_pass_function = self.do_pass_sam2_memory_attention
        if self.training:
            torch.set_grad_enabled(True)
            if self.cfg["Data"].get("synthesis") == "pretrain":
                return do_pass_function(batched_inputs, calculate_loss, synthetic_pret=True, t0_gt=True)
            return do_pass_function(batched_inputs, calculate_loss, t0_gt=t0_gt, optimizer=optimizer)
        else:
            torch.set_grad_enabled(False)
            if self.cfg["Data"].get("synthesis") == "pretrain":
                _, val_outputs =  do_pass_function(batched_inputs, calculate_loss=False, synthetic_pret=True, t0_gt=t0_gt) 
            else:
                _, val_outputs =  do_pass_function(batched_inputs, calculate_loss=False, t0_gt=t0_gt)   
            return val_outputs

