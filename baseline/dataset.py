from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import camera_box_to_lidar, draw_gaussian, map_class, parse_calib_tr_velo_to_cam, parse_label_line


class VodRadarDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str):
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg["dataset"]["root"]).resolve()
        self.class_names = cfg["dataset"]["class_names"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.class_map = cfg["dataset"]["class_map"]
        self.point_dim = int(cfg["dataset"].get("point_dim", 7))

        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = cfg["dataset"]["point_cloud_range"]
        self.vx, self.vy = cfg["dataset"]["voxel_size_xy"]
        self.grid_w = int(round((self.x_max - self.x_min) / self.vx))
        self.grid_h = int(round((self.y_max - self.y_min) / self.vy))

        split_file = self.data_root / "lidar" / "ImageSets" / f"{split}.txt"
        self.ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]

        self.radar_dir = self.data_root / "radar_5frames" / "training" / "velodyne"
        self.label_dir = self.data_root / "lidar" / "training" / "label_2"
        self.calib_dir = self.data_root / "lidar" / "training" / "calib"

    def __len__(self) -> int:
        return len(self.ids)

    def _load_points(self, sample_id: str) -> np.ndarray:
        bin_path = self.radar_dir / f"{sample_id}.bin"
        points = np.fromfile(bin_path, dtype=np.float32)
        if points.size % self.point_dim != 0:
            raise ValueError(f"Point dim mismatch in {bin_path}, got {points.size} floats")
        points = points.reshape(-1, self.point_dim)

        mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] < self.z_max)
        )
        points = points[mask]

        max_points = int(self.cfg["dataset"].get("max_points_per_sample", 0))
        if max_points > 0 and points.shape[0] > max_points:
            idx = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[idx]

        return points

    def _points_to_bev(self, points: np.ndarray) -> np.ndarray:
        bev_channels = int(self.cfg["model"]["in_channels"])
        bev = np.zeros((bev_channels, self.grid_h, self.grid_w), dtype=np.float32)
        if points.shape[0] == 0:
            return bev

        gx = ((points[:, 0] - self.x_min) / self.vx).astype(np.int32)
        gy = ((points[:, 1] - self.y_min) / self.vy).astype(np.int32)
        valid = (gx >= 0) & (gx < self.grid_w) & (gy >= 0) & (gy < self.grid_h)
        gx, gy = gx[valid], gy[valid]
        pts = points[valid]

        count = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        z_sum = np.zeros_like(count)
        rcs_sum = np.zeros_like(count)
        vr_sum = np.zeros_like(count)
        rcs_max = np.full((self.grid_h, self.grid_w), -1e9, dtype=np.float32)
        vr_min = np.full((self.grid_h, self.grid_w), 1e9, dtype=np.float32)
        vr_max = np.full((self.grid_h, self.grid_w), -1e9, dtype=np.float32)

        np.add.at(count, (gy, gx), 1.0)
        np.add.at(z_sum, (gy, gx), pts[:, 2])
        np.add.at(rcs_sum, (gy, gx), pts[:, 3])
        np.add.at(vr_sum, (gy, gx), pts[:, 4])
        np.maximum.at(rcs_max, (gy, gx), pts[:, 3])
        np.minimum.at(vr_min, (gy, gx), pts[:, 4])
        np.maximum.at(vr_max, (gy, gx), pts[:, 4])

        has = count > 0
        z_mean = np.zeros_like(count)
        rcs_mean = np.zeros_like(count)
        vr_mean = np.zeros_like(count)
        z_mean[has] = z_sum[has] / count[has]
        rcs_mean[has] = rcs_sum[has] / count[has]
        vr_mean[has] = vr_sum[has] / count[has]

        rcs_max[~has] = 0.0
        vr_min[~has] = 0.0
        vr_max[~has] = 0.0

        bev[0] = np.log1p(count)
        bev[1] = z_mean
        bev[2] = rcs_mean
        bev[3] = vr_mean
        bev[4] = rcs_max
        bev[5] = vr_min
        bev[6] = vr_max
        return bev

    def _load_gt_boxes(self, sample_id: str) -> List[Tuple[int, np.ndarray]]:
        label_path = self.label_dir / f"{sample_id}.txt"
        calib_path = self.calib_dir / f"{sample_id}.txt"
        t_velo_to_cam = parse_calib_tr_velo_to_cam(calib_path)
        gts: List[Tuple[int, np.ndarray]] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            box_cam = parse_label_line(line)
            cls_name = map_class(box_cam["name"], self.class_map)
            if cls_name == "":
                continue
            if cls_name not in self.class_to_idx:
                continue
            box_lidar = camera_box_to_lidar(box_cam, t_velo_to_cam)
            x, y, z, l, w, h, yaw = box_lidar.tolist()
            if not (self.x_min <= x < self.x_max and self.y_min <= y < self.y_max):
                continue
            gts.append((self.class_to_idx[cls_name], np.array([x, y, z, l, w, h, yaw], dtype=np.float32)))
        return gts

    def _build_targets(self, gt_boxes: List[Tuple[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        num_classes = len(self.class_names)
        heatmap = np.zeros((num_classes, self.grid_h, self.grid_w), dtype=np.float32)
        reg = np.zeros((8, self.grid_h, self.grid_w), dtype=np.float32)
        reg_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for cls_idx, box in gt_boxes:
            x, y, z, l, w, h, yaw = box.tolist()
            cx = (x - self.x_min) / self.vx
            cy = (y - self.y_min) / self.vy
            ix, iy = int(cx), int(cy)
            if ix < 0 or ix >= self.grid_w or iy < 0 or iy >= self.grid_h:
                continue

            radius = int(max(1, min(4, min(l / self.vx, w / self.vy) / 2.0)))
            draw_gaussian(heatmap[cls_idx], (iy, ix), radius)

            reg[0, iy, ix] = cx - ix
            reg[1, iy, ix] = cy - iy
            reg[2, iy, ix] = z
            reg[3, iy, ix] = np.log(max(l, 1e-3))
            reg[4, iy, ix] = np.log(max(w, 1e-3))
            reg[5, iy, ix] = np.log(max(h, 1e-3))
            reg[6, iy, ix] = np.sin(yaw)
            reg[7, iy, ix] = np.cos(yaw)
            reg_mask[iy, ix] = 1.0

        return {"heatmap": heatmap, "reg": reg, "reg_mask": reg_mask}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.ids[idx]
        points = self._load_points(sample_id)
        bev = self._points_to_bev(points)
        gt_boxes = self._load_gt_boxes(sample_id)
        targets = self._build_targets(gt_boxes)

        return {
            "sample_id": sample_id,
            "bev": torch.from_numpy(bev),
            "heatmap": torch.from_numpy(targets["heatmap"]),
            "reg": torch.from_numpy(targets["reg"]),
            "reg_mask": torch.from_numpy(targets["reg_mask"]),
            "gt_boxes": gt_boxes,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "sample_id": [x["sample_id"] for x in batch],
        "bev": torch.stack([x["bev"] for x in batch], dim=0),
        "heatmap": torch.stack([x["heatmap"] for x in batch], dim=0),
        "reg": torch.stack([x["reg"] for x in batch], dim=0),
        "reg_mask": torch.stack([x["reg_mask"] for x in batch], dim=0),
        "gt_boxes": [x["gt_boxes"] for x in batch],
    }
    return out
