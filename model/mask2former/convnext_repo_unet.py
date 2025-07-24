# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from .utils import LayerNorm, GRN
from functools import partial

## available decoder
from .upsampling_blocks import ResidualBlock, UpsampleBlock
from .swin_unet import PatchEmbed, PatchExpand, FinalPatchExpand_X4, BasicLayer_up
from .pixel_decoder.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(self.depths)-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(self.depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print('[isinstance]')
            trunc_normal_(m.weight, std=.02)
            # print('[trunc_normal_]')
            nn.init.constant_(m.bias, 0)
            # print('[constant_]')

    def forward_features(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class ConvNextV2_UNet(ConvNeXtV2):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[128,256,512,1024], #[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1,
                img_size=384,#224,
                patch_size=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                decoder_type="conv", ## swin
                output_pred=False,
                cls_head=False, 
                **kwargs):
        super().__init__(in_chans, num_classes, 
                 depths, dims, 
                 drop_path_rate, head_init_scale)

        self.decoder_type = decoder_type
        # upsampling blocks
        print(f'[convnext unet] building {self.decoder_type}_decoder...')
        if self.decoder_type == "conv":
            self.bottem_up = UpsampleBlock(dims[3], dims[2])
            self.up3 = UpsampleBlock(dims[2], dims[1])
            self.up2 = UpsampleBlock(dims[1], dims[0])
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=2, padding=1),
                ResidualBlock(dims[0], dims[0]),
                nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=2, padding=1),
                ResidualBlock(dims[0], dims[0]),
            )
        elif self.decoder_type == "swin":
            print('building swin_deocder...')
            swin_depths=[2, 2, 15, 2]
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_depths))]  # stochastic depth decay rule
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0], norm_layer=norm_layer
            )
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution
            
            self.num_features = int(dims[0] * 2 ** (self.num_stages-1 - 1))
            self.norm = norm_layer(self.num_features)
            self.build_swin_deocder(dpr, img_size, patch_size)
        elif "pixel" in self.decoder_type:
            print('building pixel_decoder...')
            # self.pixel_decoder = build_pixel_decoder()
            self.pixel_decoder = MSDeformAttnPixelDecoder(in_channels=dims,  # Feature map channels from backbone stages
                                                            strides=[4, 8, 16, 32],  # Downsampling factors for each feature map
                                                            feat_channels=256,  # Channels of intermediate features
                                                            out_channels=256,  # Output feature map channels
                                                            num_outs=3,  # Number of output feature levels
                                                            norm_cfg=dict(type='GN', num_groups=32),  # Group Normalization config
                                                            act_cfg=dict(type='ReLU'))  ## fixed temporaly
            if self.decoder_type == "pixelup":
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(256, dims[0], kernel_size=4, stride=2, padding=1),
                    ResidualBlock(dims[0], dims[0]),
                    nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=2, padding=1),
                    ResidualBlock(dims[0], dims[0]),
                )
        # self.apply(self._init_weights)
        # --------------------------------------------------------------------------
        # output
        self.output_pred = output_pred
        self.cls_head = cls_head
        if self.output_pred:
            if self.decoder_type == "pixel":
                dims = [256]*4
            self.out_conv = nn.Conv2d(dims[0], 1, kernel_size=1, padding=0)
            self.sigmoid = nn.Sigmoid()
            if self.cls_head:
                self.class_embed = nn.Linear(dims[-1]*img_size*img_size//32//32, 1)
        # --------------------------------------------------------------------------
        del self.norm, self.head

    def forward_features(self, x):
        latents_out = []
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            latents_out.append(x)
            # print('[enc]',x.shape)
        return latents_out

    def upward_features(self, latents_out):
        """latents_out: list of encoder output"""
        if self.decoder_type == "conv":
            # multi-scale features for Mask2Former Transformer Decoder
            multi_scale_features = []
            multi_scale_features.append(latents_out[3])  # [N, dims[3], H//32, W//32]

            # upsampling
            x = self.bottem_up(latents_out[3], latents_out[2])
            multi_scale_features.append(x)  # [N, dims[2], H//16, W//16]

            x = self.up3(x, latents_out[1])
            multi_scale_features.append(x)  # [N, dims[1], H//8, W//8]

            x = self.up2(x, latents_out[0])

            mask_features = self.up1(x)  # [N, dims[0], H, W]
        
        elif self.decoder_type == "swin":
            x, multi_scale_features = self.forward_up_features(latents_out[-1], latents_out[:-1])
            mask_features = self.up_x4(x)    

        elif "pixel" in self.decoder_type:
            mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(latents_out)
            if self.decoder_type == "pixelup":
                mask_features = self.up(mask_features)
            # for ms in multi_scale_features:
            #     print('[ms]', ms.shape)
        return multi_scale_features, mask_features
    
    def forward(self, x):
        """
        Output:
            - multi_scale_features: for cross-attention in Mask2Former Transformer Decoder
            - mask_features: for the final mask prediction
        """
        
        latents_out = self.forward_features(x)
        multi_scale_features, mask_features = self.upward_features(latents_out)
        
        if self.output_pred:
            if self.cls_head:
                flatten_bottom = torch.flatten(multi_scale_features[0], start_dim=1)
                outputs_class = self.class_embed(flatten_bottom)
                outputs_class = outputs_class.sigmoid()
            else:
                outputs_class = None
            mask_features = self.out_conv(mask_features)
            out = {
                "pred_class": outputs_class,
                # "pred_cals": predictions_cals,
                "pred_masks": self.sigmoid(mask_features),  ## [B, fQ, H, W]
            }
            return out    
        return multi_scale_features, mask_features
    ##############################
    ## Swin transformer decoder ##
    ##############################
    
    def latent_reshape(self, latents):
        """
        input:
        latents: (N, L, D)
        return:
        latents: (N, D, H', W')
        """
        h = w = int(latents.shape[1] ** 0.5)
        latents = latents.reshape(shape=(latents.shape[0], h, w, latents.shape[2]))  # [N, H', W', D]

        # reshape to (N, D, H', W')
        latents = latents.permute(0, 3, 1, 2)

        return latents
    
    def forward_up_features(self, x, x_downsample):
        multi_scale_features = []
        # print("\nforward_up_features")
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                bottom = x  # [N, dims[3], H//32, W//32]
                multi_scale_features.append(bottom)
                # print("\t",bottom.shape)
                x = x.flatten(2).permute(0, 2, 1).contiguous()   ## [B, H//32*W//32, dims[3]]
                x = layer_up(x)
                # print(inx, x.shape)
            else:
                x_down = x_downsample[3-inx].flatten(2).permute(0, 2, 1).contiguous()
                x = torch.cat([x,x_down],-1)
                x = self.concat_back_dim[inx](x)
                if inx < 3:
                    reshape_x = self.latent_reshape(x)
                    multi_scale_features.append(reshape_x)  # [N, dims[2], H//16, W//16]  # [N, dims[1], H//8, W//8]
                    # print("\t",reshape_x.shape)
                x = layer_up(x)
                # print(inx, x.shape)
        
        x = self.norm_up(x)  # B L C  # [N, H//4* W//4, dims[0]]
        return x, multi_scale_features
    
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        x = self.up(x)
        # print("self.up", x.shape)  
        x = x.view(B,4*H,4*W,-1)
        x = x.permute(0,3,1,2) #B,C,H,W
        # print("self.up", x.shape)  

        return x

    def build_swin_deocder(self, dpr, img_size, patch_size,
                            dims=[72, 144, 288, 576],
                            depths=[2, 2, 2, 2],  ##[2,2,15,2]
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            norm_layer=nn.LayerNorm):
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_stages-1):
            concat_linear = nn.Linear(2*dims[self.num_stages-1-i_layer], dims[self.num_stages-1-i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(self.patches_resolution[0] // (2 ** (self.num_stages-1-i_layer)),
                self.patches_resolution[1] // (2 ** (self.num_stages-1-i_layer))), dim=dims[self.num_stages-1-i_layer], dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=dims[self.num_stages-1-i_layer],
                                input_resolution=(self.patches_resolution[0] // (2 ** (self.num_stages-1-i_layer)),
                                                    self.patches_resolution[1] // (2 ** (self.num_stages-1-i_layer))),
                                depth=depths[(self.num_stages-1-i_layer)],
                                num_heads=num_heads[(self.num_stages-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=4.0,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_stages-1-i_layer)]):sum(depths[:(self.num_stages-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_stages- 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(dims[0])

        self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=dims[0])
        #     self.output = nn.Conv2d(in_channels=dims[0],out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)



def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model