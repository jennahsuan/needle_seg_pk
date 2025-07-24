# Meta Architecture for the U-Net with Mask2Former Decoder

import torch
import torch.nn as nn
import math
import os
import pkg_resources
from torch.nn import functional as F

from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


class Mask2Former(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.freeze_backbone = cfg["Train"].get("freeze_backbone", False)
        # image size
        self.img_size = cfg["Model"]["image_size"]

        # Choose the encoder type of U-Net backbone
        encoder_type = cfg["Model"]["unet_backbone"]["encoder_type"]
        if encoder_type == "TransNeXt-Tiny":
            self.embed_dims = cfg["Model"]["unet_backbone"]["transnext_tiny_params"]["embed_dims"]
            self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["transnext_tiny_params"]["pretrained_path"]
        elif encoder_type == "Swin-Small":
            self.embed_dims = cfg["Model"]["unet_backbone"]["swin_small_params"]["embed_dims"]
            self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["swin_small_params"]["pretrained_path"]
        elif encoder_type == "ConvNeXt":
            self.embed_dims = cfg["Model"]["unet_backbone"]["convnext_tiny_params"]["embed_dims"]
            self.unet_backbone_pretrained_path = cfg["Model"]["unet_backbone"]["convnext_tiny_params"]["pretrained_path"]
        else:
            raise NotImplementedError(f"U-Net with {encoder_type} encoder is not implemented.")
        decoder_type = cfg["Model"]["unet_backbone"].get("decoder_type", "conv")
        
        # U-Net backbone
        if "TransNeXt" in encoder_type:
            # if is_installed("swattention"):
            #     print("swattention package found, loading CUDA version of TransNeXt")
            #     from .transnext_unet_cuda import TransNeXtUNet
            # else:
            #     print("swattention package not found, loading PyTorch native version of TransNeXt")
            from .transnext_unet_native import TransNeXtUNet
            if self.freeze_backbone:
                rank=4
            else:
                rank=0
            self.unet_backbone = TransNeXtUNet(
                img_size=self.img_size,
                pretrain_size=self.img_size,
                embed_dims=self.embed_dims,
                decoder_type=decoder_type,
                # rank=rank
            )
        elif "Swin" in encoder_type:
            from .swin_unet import SwinUNet #, SwinSMTBackbone
            if "tiny" in self.unet_backbone_pretrained_path:
                depths= [ 2, 2, 2, 2 ]
            elif "small" in self.unet_backbone_pretrained_path:
                depths=[2, 2, 18, 2]
            self.unet_backbone = SwinUNet(
                img_size=self.img_size,
                embed_dims=self.embed_dims,
                depths=depths,
                decoder_type=decoder_type
            )
        elif "ConvNeXt" in encoder_type:
            # if os.path.exists(self.unet_backbone_pretrained_path):
            print('[Convnext] from repo')
            from .convnext_repo_unet import ConvNextV2_UNet
            depths = [3, 3, 27, 3]
            # depths = [3, 3, 27]  ## C2
            # self.embed_dims = self.embed_dims[:-1]
            self.unet_backbone = ConvNextV2_UNet(depths=depths, img_size=self.img_size, 
                                                dims=self.embed_dims, 
                                                decoder_type=decoder_type)
        print(f'{encoder_type} UNet params: ', sum(p.numel() for p in self.unet_backbone.parameters() if p.requires_grad))

        if decoder_type == "pixel":  ## deform attn
            self.trans_decoder_in_channels, self.mask_dim = [256]*3, 256  ## add input projection in transdecoder (currently not added)
            # self.trans_decoder_in_channels, self.mask_dim = 256, 256  ## transnext m2f has no projection 
        elif "pixelup" == decoder_type:
            self.trans_decoder_in_channels, self.mask_dim = [256]*3, self.embed_dims[0]
        else:
            self.trans_decoder_in_channels = self.embed_dims[1:]
            self.mask_dim = self.embed_dims[0]
        # Mask2Former Transformer Decoder
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
            ca_block=cfg["Model"]["mask2former_decoder"].get("block", "crossatt"),
            det_head = cfg["Model"]["det_head"], 
        )


    def load_pretrained_weight(self):
        if not os.path.exists(self.unet_backbone_pretrained_path) or "USFM" in self.unet_backbone_pretrained_path:
            print("No pretrain weight loaded")
            # for param in self.parameters():  # .unet_backbone
            #     param.requires_grad = True
            return
        checkpoint = torch.load(self.unet_backbone_pretrained_path)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if "state_dict" in checkpoint:  ## 512 pretrain weight ## rename the keys
            checkpoint = checkpoint["state_dict"]
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('backbone.', '')] = checkpoint.pop(key)
            # for key in list(checkpoint.keys()):
            #     if "pixel_decoder" in key:
            #         checkpoint[key.replace('.conv.', '.0.').replace('.gn.', '.1.')] = checkpoint.pop(key)
            # for key in list(checkpoint.keys()):
            #     if "decode_head." in key and ("pixel" not in key and "transformer_decoder" not in key):
            #         checkpoint.pop(key,None)
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('decode_head.', '')] = checkpoint.pop(key)
            
        self.unet_backbone.load_state_dict(checkpoint, strict=False)
        # for name, _ in self.unet_backbone.named_parameters():
        #     if name not in checkpoint.keys():
        #         print('\tno ', name)
        print(f"Encoder Pretrained weight loaded!")
        if self.freeze_backbone:
            for param in self.unet_backbone.parameters():
                param.requires_grad = False
            for param in self.unet_backbone.bottem_up.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.up3.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.up2.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.up1.parameters():
                param.requires_grad = True
            print("Freeze UNet backbone in M2F")
        

    def forward(self, x:torch.tensor):
        multi_scale_features, mask_features = self.unet_backbone(x)
        out, _ = self.transformer_decoder(multi_scale_features, mask_features)
        ### dict("pred_masks","aux_outputs","instance_embed")

        ## https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py#L222
        ## if use pixel decoder, then predictions_cals[-1] has only 1/4 resolution
        if  out["pred_masks"].shape[-1] != x.shape[-1]:  # not self.training and
            out["origin_pred_masks"] = out["pred_masks"]
            out["pred_masks"] = F.interpolate(
                out["pred_masks"],
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if self.training and out.get("aux_outputs"):
                for aux_output in out["aux_outputs"]:  # auxiliary masks
                    aux_output["pred_masks"] = F.interpolate(
                        aux_output["pred_masks"],
                        size=(x.shape[-2], x.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )
            # out["pred_masks"] = out["pred_masks"].sigmoid()  ## check if this is needy
        return out