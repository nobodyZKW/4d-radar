from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from pcdet.models.backbones_3d.vfe.vfe_template import VFETemplate


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = _scatter_sum(src, index, dim_size)
    cnt = src.new_zeros((dim_size, 1))
    ones = src.new_ones((src.shape[0], 1))
    cnt.index_add_(0, index, ones)
    return out / cnt.clamp(min=1.0)


def _scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = src.new_full((dim_size, src.shape[1]), float("-inf"))
        idx = index.unsqueeze(1).expand(-1, src.shape[1])
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
        # Avoid in-place edit on scatter_reduce result to keep autograd graph valid.
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    # Fallback for older PyTorch.
    out = src.new_zeros((dim_size, src.shape[1]))
    for i in range(dim_size):
        mask = index == i
        if mask.any():
            out[i] = src[mask].max(dim=0).values
    return out


def _build_mlp(in_channels: int, channels: List[int], use_norm: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    c_in = in_channels
    for c_out in channels:
        layers.append(nn.Linear(c_in, c_out, bias=not use_norm))
        if use_norm:
            layers.append(nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01))
        layers.append(nn.ReLU(inplace=True))
        c_in = c_out
    return nn.Sequential(*layers)


class Radar7PillarVFE(VFETemplate):
    """Dynamic pillar VFE for 7D radar points:
    [x, y, z, rcs, v_r, v_r_comp, time].
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = bool(self.model_cfg.get("USE_NORM", True))
        self.use_xyz = bool(self.model_cfg.get("USE_XYZ", True))
        self.use_rcs = bool(self.model_cfg.get("USE_RCS", True))
        self.use_vr = bool(self.model_cfg.get("USE_VR", True))
        self.use_vr_comp = bool(self.model_cfg.get("USE_VR_COMP", True))
        self.use_time = bool(self.model_cfg.get("USE_TIME", True))
        self.use_elevation = bool(self.model_cfg.get("USE_ELEVATION", True))
        self.use_distance = bool(self.model_cfg.get("USE_DISTANCE", True))

        self.geo_mlp_channels = list(self.model_cfg.get("GEO_MLP", [32, 32]))
        self.motion_mlp_channels = list(self.model_cfg.get("MOTION_MLP", [32, 32]))
        self.fuse_mlp_channels = list(self.model_cfg.get("FUSE_MLP", [64]))

        geo_in = 0
        if self.use_xyz:
            geo_in += 3
        if self.use_rcs:
            geo_in += 1
        geo_in += 3  # cluster offset
        geo_in += 2  # pillar center offset (x, y)
        if self.use_elevation:
            geo_in += 1  # pillar z offset
        if self.use_distance:
            geo_in += 1

        motion_in = 0
        if self.use_vr:
            motion_in += 1
        if self.use_vr_comp:
            motion_in += 1
        if self.use_time:
            motion_in += 1
        motion_in += 1  # radial range

        if geo_in <= 0:
            geo_in = 1
        if motion_in <= 0:
            motion_in = 1

        self.geo_mlp = _build_mlp(geo_in, self.geo_mlp_channels, self.use_norm)
        self.motion_mlp = _build_mlp(motion_in, self.motion_mlp_channels, self.use_norm)

        geo_out = self.geo_mlp_channels[-1] if self.geo_mlp_channels else geo_in
        motion_out = self.motion_mlp_channels[-1] if self.motion_mlp_channels else motion_in
        fuse_in = (geo_out + motion_out) * 2  # max + mean pooling per branch

        self.fuse_mlp = _build_mlp(fuse_in, self.fuse_mlp_channels, self.use_norm)
        self.output_channels = self.fuse_mlp_channels[-1] if self.fuse_mlp_channels else fuse_in

        self.voxel_x = float(voxel_size[0])
        self.voxel_y = float(voxel_size[1])
        self.voxel_z = float(voxel_size[2])

        self.x_offset = self.voxel_x / 2 + float(point_cloud_range[0])
        self.y_offset = self.voxel_y / 2 + float(point_cloud_range[1])
        self.z_offset = self.voxel_z / 2 + float(point_cloud_range[2])

        self.scale_xy = int(grid_size[0] * grid_size[1])
        self.scale_y = int(grid_size[1])

        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.int32), persistent=False)
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)

    def get_output_feature_dim(self):
        return self.output_channels

    def _empty_output(self, batch_dict, device, dtype):
        batch_size = int(batch_dict.get("batch_size", 1))
        voxel_features = torch.zeros((batch_size, self.output_channels), dtype=dtype, device=device)
        voxel_coords = torch.zeros((batch_size, 4), dtype=torch.int32, device=device)
        voxel_coords[:, 0] = torch.arange(batch_size, device=device, dtype=torch.int32)
        batch_dict["voxel_features"] = voxel_features
        batch_dict["pillar_features"] = voxel_features
        batch_dict["voxel_coords"] = voxel_coords
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        # points: [batch_idx, x, y, z, rcs, v_r, v_r_comp, time]
        points = batch_dict["points"]
        if points.shape[0] == 0:
            return self._empty_output(batch_dict, points.device, points.dtype)

        pts_xy = points[:, [1, 2]]
        coords_xy = torch.floor((pts_xy - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        valid = ((coords_xy >= 0) & (coords_xy < self.grid_size[[0, 1]])).all(dim=1)
        if not valid.any():
            return self._empty_output(batch_dict, points.device, points.dtype)

        points = points[valid]
        coords_xy = coords_xy[valid]

        pts_xyz = points[:, 1:4].contiguous()
        pts_rcs = points[:, 4:5] if points.shape[1] > 4 else points.new_zeros((points.shape[0], 1))
        pts_vr = points[:, 5:6] if points.shape[1] > 5 else points.new_zeros((points.shape[0], 1))
        pts_vr_comp = points[:, 6:7] if points.shape[1] > 6 else points.new_zeros((points.shape[0], 1))
        pts_time = points[:, 7:8] if points.shape[1] > 7 else points.new_zeros((points.shape[0], 1))

        merge_coords = (
            points[:, 0].int() * self.scale_xy
            + coords_xy[:, 0] * self.scale_y
            + coords_xy[:, 1]
        )
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True)
        num_pillars = int(unq_coords.shape[0])

        pts_mean = _scatter_mean(pts_xyz, unq_inv, num_pillars)
        f_cluster = pts_xyz - pts_mean[unq_inv]

        f_center = torch.zeros_like(pts_xyz)
        f_center[:, 0] = pts_xyz[:, 0] - (coords_xy[:, 0].to(pts_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = pts_xyz[:, 1] - (coords_xy[:, 1].to(pts_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = pts_xyz[:, 2] - self.z_offset

        radial_range = torch.norm(pts_xyz[:, :2], p=2, dim=1, keepdim=True)

        geo_parts = []
        if self.use_xyz:
            geo_parts.append(pts_xyz)
        if self.use_rcs:
            geo_parts.append(pts_rcs)
        geo_parts.extend([f_cluster, f_center[:, :2]])
        if self.use_elevation:
            geo_parts.append(f_center[:, 2:3])
        if self.use_distance:
            geo_parts.append(torch.norm(pts_xyz, p=2, dim=1, keepdim=True))
        if len(geo_parts) == 0:
            geo_parts.append(points.new_zeros((points.shape[0], 1)))

        motion_parts = []
        if self.use_vr:
            motion_parts.append(pts_vr)
        if self.use_vr_comp:
            motion_parts.append(pts_vr_comp)
        if self.use_time:
            motion_parts.append(pts_time)
        motion_parts.append(radial_range)
        if len(motion_parts) == 0:
            motion_parts.append(points.new_zeros((points.shape[0], 1)))

        geo_in = torch.cat(geo_parts, dim=1)
        motion_in = torch.cat(motion_parts, dim=1)

        geo_point = self.geo_mlp(geo_in)
        motion_point = self.motion_mlp(motion_in)

        geo_pillar = torch.cat([
            _scatter_max(geo_point, unq_inv, num_pillars),
            _scatter_mean(geo_point, unq_inv, num_pillars),
        ], dim=1)
        motion_pillar = torch.cat([
            _scatter_max(motion_point, unq_inv, num_pillars),
            _scatter_mean(motion_point, unq_inv, num_pillars),
        ], dim=1)

        pillar_features = torch.cat([geo_pillar, motion_pillar], dim=1)
        if len(self.fuse_mlp) > 0:
            pillar_features = self.fuse_mlp(pillar_features)

        unq_coords = unq_coords.int()
        batch_idx = unq_coords // self.scale_xy
        x_idx = (unq_coords % self.scale_xy) // self.scale_y
        y_idx = unq_coords % self.scale_y

        voxel_coords = torch.stack(
            (
                batch_idx,
                torch.zeros_like(batch_idx),
                y_idx,
                x_idx,
            ),
            dim=1,
        ).int()

        batch_dict["voxel_features"] = pillar_features
        batch_dict["pillar_features"] = pillar_features
        batch_dict["voxel_coords"] = voxel_coords
        return batch_dict
