from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RadarBaselineNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 1),
            ConvBNReLU(32, 32, 3, 1),
        )
        self.down1 = nn.Sequential(
            ConvBNReLU(32, 64, 3, 2),
            ConvBNReLU(64, 64, 3, 1),
        )
        self.down2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2),
            ConvBNReLU(128, 128, 3, 1),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.fuse1 = ConvBNReLU(64 + 64, 64, 3, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.fuse2 = ConvBNReLU(32 + 32, 32, 3, 1)

        self.head_hm = nn.Sequential(
            ConvBNReLU(32, 32, 3, 1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self.head_reg = nn.Sequential(
            ConvBNReLU(32, 32, 3, 1),
            nn.Conv2d(32, 8, kernel_size=1),
        )

        # Keep early logits biased to low confidence for stable start.
        nn.init.constant_(self.head_hm[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        u1 = self.up1(x2)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        f1 = self.fuse1(torch.cat([u1, x1], dim=1))

        u2 = self.up2(f1)
        if u2.shape[-2:] != x0.shape[-2:]:
            u2 = F.interpolate(u2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse2(torch.cat([u2, x0], dim=1))

        return {"heatmap": self.head_hm(feat), "reg": self.head_reg(feat)}
