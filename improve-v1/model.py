from typing import Dict, List

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


class RadarCenterPointV1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_motion_branch: bool = True,
        geometry_channels: List[int] | None = None,
        motion_channels: List[int] | None = None,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.use_motion_branch = bool(use_motion_branch)

        if geometry_channels is None:
            geometry_channels = list(range(min(7, in_channels)))
        if motion_channels is None:
            motion_channels = [i for i in range(in_channels) if i not in geometry_channels]
        if len(geometry_channels) == 0:
            geometry_channels = list(range(in_channels))
        if len(motion_channels) == 0:
            motion_channels = [0]

        self.register_buffer("geometry_channels", torch.tensor(geometry_channels, dtype=torch.long), persistent=False)
        self.register_buffer("motion_channels", torch.tensor(motion_channels, dtype=torch.long), persistent=False)

        if self.use_motion_branch:
            self.geo_stem = nn.Sequential(
                ConvBNReLU(len(geometry_channels), base_channels, 3, 1),
                ConvBNReLU(base_channels, base_channels, 3, 1),
            )
            self.motion_stem = nn.Sequential(
                ConvBNReLU(len(motion_channels), base_channels, 3, 1),
                ConvBNReLU(base_channels, base_channels, 3, 1),
            )
            stem_in = base_channels * 2
        else:
            self.stem = nn.Sequential(
                ConvBNReLU(in_channels, base_channels, 3, 1),
                ConvBNReLU(base_channels, base_channels, 3, 1),
            )
            stem_in = base_channels

        self.stem_fuse = ConvBNReLU(stem_in, base_channels, 3, 1)

        self.down1 = nn.Sequential(
            ConvBNReLU(base_channels, base_channels * 2, 3, 2),
            ConvBNReLU(base_channels * 2, base_channels * 2, 3, 1),
        )
        self.down2 = nn.Sequential(
            ConvBNReLU(base_channels * 2, base_channels * 4, 3, 2),
            ConvBNReLU(base_channels * 4, base_channels * 4, 3, 1),
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.fuse1 = ConvBNReLU(base_channels * 4, base_channels * 2, 3, 1)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.fuse2 = ConvBNReLU(base_channels * 2, base_channels, 3, 1)

        self.head_hm = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, num_classes, 1))
        self.head_offset = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, 2, 1))
        self.head_z = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, 1, 1))
        self.head_size = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, 3, 1))
        self.head_yaw = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, 2, 1))
        # Keep velocity head always present; enable/disable loss through config.
        self.head_vel = nn.Sequential(ConvBNReLU(base_channels, base_channels, 3, 1), nn.Conv2d(base_channels, 2, 1))

        nn.init.constant_(self.head_hm[-1].bias, -2.19)

    def _build_stem(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_motion_branch:
            geo = x.index_select(1, self.geometry_channels.clamp(max=x.shape[1] - 1))
            mot = x.index_select(1, self.motion_channels.clamp(max=x.shape[1] - 1))
            geo_f = self.geo_stem(geo)
            mot_f = self.motion_stem(mot)
            x0 = self.stem_fuse(torch.cat([geo_f, mot_f], dim=1))
            return x0
        return self.stem_fuse(self.stem(x))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self._build_stem(x)
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

        return {
            "heatmap": self.head_hm(feat),
            "offset": self.head_offset(feat),
            "z": self.head_z(feat),
            "size": self.head_size(feat),
            "yaw": self.head_yaw(feat),
            "vel": self.head_vel(feat),
        }

