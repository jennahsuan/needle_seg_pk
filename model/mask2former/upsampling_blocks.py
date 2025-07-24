# Upsampling blocks for the U-Net architecture

import torch
import torch.nn as nn


# single convolutional block with GroupNorm and GELU activation (single yellow block in the architecture diagram)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.dropout.p > 0.0:
            x = self.dropout(x)
        return x


# 2 BasicBlock with residual connection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = BasicBlock(in_channels, out_channels)
        self.block2 = BasicBlock(out_channels, out_channels)
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out + self.res(x)
        return out


# upsampling block (green block followed by concatenation and 2 yellow blocks in the architecture diagram)
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x
