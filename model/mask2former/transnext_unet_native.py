# TransNeXt U-Net Model
# https://github.com/DaiShiResearch/TransNeXt/blob/main/segmentation/mask2former/transnext_native.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch.nn import functional as F

from .upsampling_blocks import ResidualBlock, UpsampleBlock
from .pixel_decoder.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = (
        torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))
    )

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(
        torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device),
        window_size,
        dilation=1,
        padding=(window_size // 2, window_size // 2),
        stride=1,
    )
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(
        self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, is_extrapolation=False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size**2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = (
            self.unfold(kv_local).reshape(B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        )
        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(
            padding_mask, float("-inf")
        )

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = (
                self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, N, pool_len)
            )
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = (
                self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
                .transpose(0, 1)[:, relative_pos_index.view(-1)]
                .view(-1, self.trained_len, self.trained_pool_len)
            )

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode="bilinear")
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len, self.trained_H, self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode="bilinear").reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        if self.is_extrapolation:
            # Use MLP to generate continuous relative positional bias
            rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, N, N)
        else:
            # Use MLP to generate continuous relative positional bias
            rel_bias = (
                self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table)))
                .transpose(0, 1)[:, relative_pos_index.view(-1)]
                .view(-1, self.trained_len, self.trained_len)
            )
            # bilinear interpolation:
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode="bilinear")
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(-1, N, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode="bilinear").reshape(-1, N, N).transpose(-1, -2)

        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale) @ F.normalize(k, dim=-1).transpose(
            -2, -1
        ) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        input_resolution,
        window_size=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        is_extrapolation=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(
                dim, input_resolution, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_extrapolation=is_extrapolation
            )
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation,
            )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


""" TransNeXt U-Net """


class TransNeXtUNet(nn.Module):
    """
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation can accept multi-scale inputs,
    it is recommended to set the "img size" parameter to a value close to the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    """

    def __init__(
        self,
        img_size=224,
        pretrain_size=224,
        window_size=[3, 3, 3, None],
        patch_size=4,
        in_chans=3,
        embed_dims=[72, 144, 288, 576],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 15, 2],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        is_extrapolation=False,
        decoder_type="conv", ## swin
        output_pred=False,
        cls_head=False
    ):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))),
                )

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(
                patch_size=patch_size * 2 - 1 if i == 0 else 3,
                stride=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        input_resolution=to_2tuple(img_size // (2 ** (i + 2))),
                        window_size=window_size[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        is_extrapolation=is_extrapolation,
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.decoder_type = decoder_type
        # upsampling blocks
        if self.decoder_type == "conv":
            self.bottem_up = UpsampleBlock(embed_dims[3], embed_dims[2])
            self.up3 = UpsampleBlock(embed_dims[2], embed_dims[1])
            self.up2 = UpsampleBlock(embed_dims[1], embed_dims[0])
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=4, stride=2, padding=1),
                ResidualBlock(embed_dims[0], embed_dims[0]),
                nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=4, stride=2, padding=1),
                ResidualBlock(embed_dims[0], embed_dims[0]),
            )
        elif self.decoder_type == "swin":
            print('building swin_deocder...')
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer
            )
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution
            
            self.num_features = int(embed_dims[0] * 2 ** (self.num_stages-1 - 1))
            self.norm = norm_layer(self.num_features)
            self.build_swin_deocder(dpr, img_size, patch_size)
        elif "pixel" in self.decoder_type:
            print('building pixel_decoder...')
            self.pixel_decoder = MSDeformAttnPixelDecoder(in_channels=[72, 144, 288, 576],  # Feature map channels from backbone stages
                                                            strides=[4, 8, 16, 32],  # Downsampling factors for each feature map
                                                            feat_channels=256,  # Channels of intermediate features
                                                            out_channels=256,  # Output feature map channels
                                                            num_outs=3,  # Number of output feature levels
                                                            norm_cfg=dict(type='GN', num_groups=32),  # Group Normalization config
                                                            act_cfg=dict(type='ReLU'))  ## fixed temporaly
            if self.decoder_type == "pixelup":
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(256, embed_dims[0], kernel_size=4, stride=2, padding=1),
                    ResidualBlock(embed_dims[0], embed_dims[0]),
                    nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=4, stride=2, padding=1),
                    ResidualBlock(embed_dims[0], embed_dims[0]),
                )
        for n, m in self.named_modules():
            self._init_weights(m, n)

        # --------------------------------------------------------------------------
        # output
        self.output_pred = output_pred
        self.cls_head = cls_head
        if self.output_pred:
            self.out_conv = nn.Conv2d(embed_dims[0], 1, kernel_size=1, padding=0)
            self.sigmoid = nn.Sigmoid()
            if self.cls_head:
                self.class_embed = nn.Linear(embed_dims[-1]*img_size*img_size//32//32, 1)
        # --------------------------------------------------------------------------


    def _init_weights(self, m: nn.Module, name: str = ""):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"query_embedding", "relative_pos_bias_local", "cpb", "temperature"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=(H, W),
                    key_size=(H // sr_ratio, W // sr_ratio),
                    pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))),
                    device=x.device,
                )
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))
                    padding_mask = None
            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        """
        Output:
            - multi_scale_features: for cross-attention in Mask2Former Transformer Decoder
            - mask_features: for the final mask prediction
        """
        latents_out = self.forward_features(x)

        if self.decoder_type == "conv":
            # multi-scale features for Mask2Former Transformer Decoder
            multi_scale_features = []
            multi_scale_features.append(latents_out[3])  # [N, embed_dims[3], H//32, W//32]

            # upsampling
            x = self.bottem_up(latents_out[3], latents_out[2])
            multi_scale_features.append(x)  # [N, embed_dims[2], H//16, W//16]

            x = self.up3(x, latents_out[1])
            multi_scale_features.append(x)  # [N, embed_dims[1], H//8, W//8]

            x = self.up2(x, latents_out[0])

            mask_features = self.up1(x)  # [N, embed_dims[0], H, W]
        
        elif self.decoder_type == "swin":
            x, multi_scale_features = self.forward_up_features(latents_out[-1], latents_out[:-1])
            mask_features = self.up_x4(x)    

        elif "pixel" in self.decoder_type:
            mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(latents_out)
            if self.decoder_type == "pixelup":
                mask_features = self.up(mask_features)
                
        if self.output_pred:
            if self.cls_head:
                flatten_bottom = torch.flatten(multi_scale_features[0], start_dim=1)
                outputs_class = self.class_embed(flatten_bottom)
                outputs_class = outputs_class.sigmoid()
            else:
                outputs_class = None
            mask_features = self.out_conv(mask_features)
            out = {
                "pred_logits": outputs_class,
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
                bottom = x  # [N, embed_dims[3], H//32, W//32]
                multi_scale_features.append(bottom)
                # print("\t",bottom.shape)
                x = x.flatten(2).permute(0, 2, 1).contiguous()   ## [B, H//32*W//32, embed_dims[3]]
                x = layer_up(x)
                # print(inx, x.shape)
            else:
                x_down = x_downsample[3-inx].flatten(2).permute(0, 2, 1).contiguous()
                x = torch.cat([x,x_down],-1)
                x = self.concat_back_dim[inx](x)
                if inx < 3:
                    reshape_x = self.latent_reshape(x)
                    multi_scale_features.append(reshape_x)  # [N, embed_dims[2], H//16, W//16]  # [N, embed_dims[1], H//8, W//8]
                    # print("\t",reshape_x.shape)
                x = layer_up(x)
                # print(inx, x.shape)
        
        x = self.norm_up(x)  # B L C  # [N, H//4* W//4, embed_dims[0]]
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
                            embed_dims=[72, 144, 288, 576],
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
            concat_linear = nn.Linear(2*embed_dims[self.num_stages-1-i_layer], embed_dims[self.num_stages-1-i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(self.patches_resolution[0] // (2 ** (self.num_stages-1-i_layer)),
                self.patches_resolution[1] // (2 ** (self.num_stages-1-i_layer))), dim=embed_dims[self.num_stages-1-i_layer], dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=embed_dims[self.num_stages-1-i_layer],
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
        self.norm_up= norm_layer(embed_dims[0])

        self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dims[0])
        #     self.output = nn.Conv2d(in_channels=embed_dims[0],out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

