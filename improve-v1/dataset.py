from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .utils import camera_box_to_lidar, draw_gaussian, map_class, parse_calib_tr_velo_to_cam, parse_label_line
except ImportError:
    from utils import camera_box_to_lidar, draw_gaussian, map_class, parse_calib_tr_velo_to_cam, parse_label_line


def _normalize_angle(rad: np.ndarray) -> np.ndarray:
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


class VodRadarDatasetV1(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str, training: bool = False):
        self.cfg = cfg
        self.split = split
        self.training = bool(training)
        self.data_root = Path(cfg["dataset"]["root"]).resolve()
        self.class_names = cfg["dataset"]["class_names"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.class_map = cfg["dataset"]["class_map"]
        self.point_dim = int(cfg["dataset"].get("point_dim", 7))

        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = cfg["dataset"]["point_cloud_range"]
        self.vx, self.vy = cfg["dataset"]["voxel_size_xy"]
        self.grid_w = int(round((self.x_max - self.x_min) / self.vx))
        self.grid_h = int(round((self.y_max - self.y_min) / self.vy))

        self.bev_feature_set = str(cfg["dataset"].get("bev_feature_set", "extended16")).lower()
        self.num_input_channels = int(cfg["dataset"].get("num_input_channels", 16))
        self.feature_indices = cfg["dataset"].get(
            "feature_indices", {"rcs": 3, "doppler": 4, "comp_doppler": 5, "time": 6}
        )

        aug_cfg = cfg["dataset"].get("augment", {})
        self.aug_enable = bool(aug_cfg.get("enable", False)) and self.training
        self.flip_prob = float(aug_cfg.get("flip_prob", 0.0))
        self.scale_min, self.scale_max = [float(x) for x in aug_cfg.get("scale_range", [1.0, 1.0])]
        rot_deg = aug_cfg.get("rot_range_deg", [0.0, 0.0])
        self.rot_min = float(rot_deg[0]) * np.pi / 180.0
        self.rot_max = float(rot_deg[1]) * np.pi / 180.0
        self.point_dropout_prob = float(aug_cfg.get("point_dropout_prob", 0.0))

        self.velocity_target_mode = str(cfg["dataset"].get("velocity_target_mode", "none")).lower()
        self.velocity_target_min_points = int(cfg["dataset"].get("velocity_target_min_points", 4))

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
        return points

    def _clip_points_range(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points
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
            if cls_name == "" or cls_name not in self.class_to_idx:
                continue
            box_lidar = camera_box_to_lidar(box_cam, t_velo_to_cam)
            x, y, z, l, w, h, yaw = box_lidar.tolist()
            if not (self.x_min <= x < self.x_max and self.y_min <= y < self.y_max):
                continue
            # v1 keeps velocity slot as optional target: [vx, vy].
            ext_box = np.array([x, y, z, l, w, h, yaw, np.nan, np.nan], dtype=np.float32)
            gts.append((self.class_to_idx[cls_name], ext_box))
        return gts

    def _apply_augment(
        self, points: np.ndarray, gt_boxes: List[Tuple[int, np.ndarray]]
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        if not self.aug_enable:
            return points, gt_boxes
        if points.shape[0] == 0 and len(gt_boxes) == 0:
            return points, gt_boxes

        boxes_arr = np.array([b for _, b in gt_boxes], dtype=np.float32) if gt_boxes else np.zeros((0, 9), np.float32)
        cls_arr = [c for c, _ in gt_boxes]

        # 1) left-right flip in lidar y axis
        if np.random.rand() < self.flip_prob:
            if points.shape[0] > 0:
                points[:, 1] *= -1.0
            if boxes_arr.shape[0] > 0:
                boxes_arr[:, 1] *= -1.0
                boxes_arr[:, 6] *= -1.0
                valid_vel = np.isfinite(boxes_arr[:, 8])
                boxes_arr[valid_vel, 8] *= -1.0

        # 2) global scaling
        if self.scale_max > self.scale_min:
            scale = np.random.uniform(self.scale_min, self.scale_max)
        else:
            scale = self.scale_min
        if abs(scale - 1.0) > 1e-6:
            if points.shape[0] > 0:
                points[:, :3] *= scale
            if boxes_arr.shape[0] > 0:
                boxes_arr[:, :6] *= scale

        # 3) small rotation around z axis
        if self.rot_max > self.rot_min:
            ang = np.random.uniform(self.rot_min, self.rot_max)
            c = float(np.cos(ang))
            s = float(np.sin(ang))
            if points.shape[0] > 0:
                x = points[:, 0].copy()
                y = points[:, 1].copy()
                points[:, 0] = c * x - s * y
                points[:, 1] = s * x + c * y
            if boxes_arr.shape[0] > 0:
                x = boxes_arr[:, 0].copy()
                y = boxes_arr[:, 1].copy()
                boxes_arr[:, 0] = c * x - s * y
                boxes_arr[:, 1] = s * x + c * y
                boxes_arr[:, 6] = _normalize_angle(boxes_arr[:, 6] + ang)
                vel_ok = np.isfinite(boxes_arr[:, 7]) & np.isfinite(boxes_arr[:, 8])
                if np.any(vel_ok):
                    vx = boxes_arr[vel_ok, 7].copy()
                    vy = boxes_arr[vel_ok, 8].copy()
                    boxes_arr[vel_ok, 7] = c * vx - s * vy
                    boxes_arr[vel_ok, 8] = s * vx + c * vy

        # 4) point dropout
        if points.shape[0] > 0 and self.point_dropout_prob > 0.0:
            keep = np.random.rand(points.shape[0]) > self.point_dropout_prob
            if np.any(keep):
                points = points[keep]

        # Rebuild gt list and clip to point cloud range.
        new_gts: List[Tuple[int, np.ndarray]] = []
        for i, cls_idx in enumerate(cls_arr):
            box = boxes_arr[i]
            if self.x_min <= box[0] < self.x_max and self.y_min <= box[1] < self.y_max:
                new_gts.append((cls_idx, box.astype(np.float32)))
        return points, new_gts

    def _points_in_box_mask(self, points: np.ndarray, box: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        x, y, z, l, w, h, yaw = [float(v) for v in box[:7]]
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        dz = points[:, 2] - z
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return (np.abs(lx) <= l * 0.5) & (np.abs(ly) <= w * 0.5) & (np.abs(dz) <= h * 0.5)

    def _assign_velocity_target(self, points: np.ndarray, gt_boxes: List[Tuple[int, np.ndarray]]) -> None:
        mode = self.velocity_target_mode
        if mode == "none" or not gt_boxes:
            return
        comp_idx = int(self.feature_indices.get("comp_doppler", 5))
        if comp_idx >= points.shape[1]:
            return

        for i in range(len(gt_boxes)):
            cls_idx, box = gt_boxes[i]
            mask = self._points_in_box_mask(points, box)
            if int(mask.sum()) < self.velocity_target_min_points:
                continue
            comp = points[mask, comp_idx]
            if comp.shape[0] == 0:
                continue
            comp_val = float(np.median(comp))
            yaw = float(box[6])
            # v1 approximation: project compensated radial motion to heading axis.
            box[7] = comp_val * float(np.cos(yaw))
            box[8] = comp_val * float(np.sin(yaw))
            gt_boxes[i] = (cls_idx, box)

    @staticmethod
    def _safe_col(points: np.ndarray, idx: int) -> np.ndarray:
        if points.shape[1] > idx:
            return points[:, idx].astype(np.float32)
        return np.zeros((points.shape[0],), dtype=np.float32)

    def _acc_stats(self, gy: np.ndarray, gx: np.ndarray, values: np.ndarray, count: np.ndarray) -> Dict[str, np.ndarray]:
        shp = count.shape
        s = np.zeros(shp, dtype=np.float32)
        ss = np.zeros(shp, dtype=np.float32)
        vmin = np.full(shp, np.inf, dtype=np.float32)
        vmax = np.full(shp, -np.inf, dtype=np.float32)
        np.add.at(s, (gy, gx), values)
        np.add.at(ss, (gy, gx), values * values)
        np.minimum.at(vmin, (gy, gx), values)
        np.maximum.at(vmax, (gy, gx), values)

        has = count > 0
        mean = np.zeros(shp, dtype=np.float32)
        std = np.zeros(shp, dtype=np.float32)
        mean[has] = s[has] / count[has]
        var = np.zeros(shp, dtype=np.float32)
        var[has] = ss[has] / count[has] - mean[has] * mean[has]
        std[has] = np.sqrt(np.maximum(var[has], 0.0))
        vmin[~has] = 0.0
        vmax[~has] = 0.0
        return {"mean": mean, "std": std, "min": vmin, "max": vmax}

    def _points_to_bev(self, points: np.ndarray) -> np.ndarray:
        bev = np.zeros((self.num_input_channels, self.grid_h, self.grid_w), dtype=np.float32)
        if points.shape[0] == 0:
            return bev

        gx = ((points[:, 0] - self.x_min) / self.vx).astype(np.int32)
        gy = ((points[:, 1] - self.y_min) / self.vy).astype(np.int32)
        valid = (gx >= 0) & (gx < self.grid_w) & (gy >= 0) & (gy < self.grid_h)
        if not np.any(valid):
            return bev
        gx = gx[valid]
        gy = gy[valid]
        pts = points[valid]

        count = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        np.add.at(count, (gy, gx), 1.0)

        z = pts[:, 2].astype(np.float32)
        rcs = self._safe_col(pts, int(self.feature_indices.get("rcs", 3)))
        doppler = self._safe_col(pts, int(self.feature_indices.get("doppler", 4)))
        comp_doppler = self._safe_col(pts, int(self.feature_indices.get("comp_doppler", 5)))
        time = self._safe_col(pts, int(self.feature_indices.get("time", 6)))

        zs = self._acc_stats(gy, gx, z, count)
        rs = self._acc_stats(gy, gx, rcs, count)
        ds = self._acc_stats(gy, gx, doppler, count)
        cds = self._acc_stats(gy, gx, comp_doppler, count)
        ts = self._acc_stats(gy, gx, time, count)

        if self.bev_feature_set == "baseline7":
            channels = [
                np.log1p(count),
                zs["mean"],
                rs["mean"],
                ds["mean"],
                rs["max"],
                ds["min"],
                ds["max"],
            ]
        else:
            channels = [
                np.log1p(count),  # 1 count
                zs["mean"],  # 2 z_mean
                zs["max"],  # 3 z_max
                zs["std"],  # 4 z_std
                rs["mean"],  # 5 rcs_mean
                rs["max"],  # 6 rcs_max
                rs["std"],  # 7 rcs_std
                ds["mean"],  # 8 doppler_mean
                ds["min"],  # 9 doppler_min
                ds["max"],  # 10 doppler_max
                cds["mean"],  # 11 comp_doppler_mean
                cds["min"],  # 12 comp_doppler_min
                cds["max"],  # 13 comp_doppler_max
                ts["mean"],  # 14 time_mean
                ts["max"],  # 15 time_max
                ts["std"],  # 16 time_std
            ]

        if len(channels) != self.num_input_channels:
            raise ValueError(
                f"BEV channels mismatch: feature_set={self.bev_feature_set}, got={len(channels)}, "
                f"cfg.num_input_channels={self.num_input_channels}"
            )
        bev[:] = np.stack(channels, axis=0).astype(np.float32)
        return bev

    def _build_targets(self, gt_boxes: List[Tuple[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        num_classes = len(self.class_names)
        heatmap = np.zeros((num_classes, self.grid_h, self.grid_w), dtype=np.float32)
        offset = np.zeros((2, self.grid_h, self.grid_w), dtype=np.float32)
        z = np.zeros((1, self.grid_h, self.grid_w), dtype=np.float32)
        size = np.zeros((3, self.grid_h, self.grid_w), dtype=np.float32)
        yaw = np.zeros((2, self.grid_h, self.grid_w), dtype=np.float32)
        vel = np.zeros((2, self.grid_h, self.grid_w), dtype=np.float32)
        reg_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        vel_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for cls_idx, box in gt_boxes:
            x, yv, zv, l, w, h, yaw_v = [float(v) for v in box[:7]]
            cx = (x - self.x_min) / self.vx
            cy = (yv - self.y_min) / self.vy
            ix, iy = int(cx), int(cy)
            if ix < 0 or ix >= self.grid_w or iy < 0 or iy >= self.grid_h:
                continue

            radius = int(max(1, min(4, min(l / self.vx, w / self.vy) / 2.0)))
            draw_gaussian(heatmap[cls_idx], (iy, ix), radius)

            offset[0, iy, ix] = cx - ix
            offset[1, iy, ix] = cy - iy
            z[0, iy, ix] = zv
            size[0, iy, ix] = np.log(max(l, 1e-3))
            size[1, iy, ix] = np.log(max(w, 1e-3))
            size[2, iy, ix] = np.log(max(h, 1e-3))
            yaw[0, iy, ix] = np.sin(yaw_v)
            yaw[1, iy, ix] = np.cos(yaw_v)
            reg_mask[iy, ix] = 1.0

            if np.isfinite(box[7]) and np.isfinite(box[8]):
                vel[0, iy, ix] = float(box[7])
                vel[1, iy, ix] = float(box[8])
                vel_mask[iy, ix] = 1.0

        return {
            "heatmap": heatmap,
            "offset": offset,
            "z": z,
            "size": size,
            "yaw": yaw,
            "vel": vel,
            "reg_mask": reg_mask,
            "vel_mask": vel_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.ids[idx]
        points = self._load_points(sample_id)
        gt_boxes = self._load_gt_boxes(sample_id)
        points, gt_boxes = self._apply_augment(points, gt_boxes)
        points = self._clip_points_range(points)
        self._assign_velocity_target(points, gt_boxes)
        bev = self._points_to_bev(points)
        targets = self._build_targets(gt_boxes)

        gt_boxes_eval = [(cls_idx, box[:7].astype(np.float32).copy()) for cls_idx, box in gt_boxes]
        return {
            "sample_id": sample_id,
            "bev": torch.from_numpy(bev),
            "heatmap": torch.from_numpy(targets["heatmap"]),
            "offset": torch.from_numpy(targets["offset"]),
            "z": torch.from_numpy(targets["z"]),
            "size": torch.from_numpy(targets["size"]),
            "yaw": torch.from_numpy(targets["yaw"]),
            "vel": torch.from_numpy(targets["vel"]),
            "reg_mask": torch.from_numpy(targets["reg_mask"]),
            "vel_mask": torch.from_numpy(targets["vel_mask"]),
            "gt_boxes": gt_boxes_eval,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "bev": torch.stack([x["bev"] for x in batch], dim=0),
        "heatmap": torch.stack([x["heatmap"] for x in batch], dim=0),
        "offset": torch.stack([x["offset"] for x in batch], dim=0),
        "z": torch.stack([x["z"] for x in batch], dim=0),
        "size": torch.stack([x["size"] for x in batch], dim=0),
        "yaw": torch.stack([x["yaw"] for x in batch], dim=0),
        "vel": torch.stack([x["vel"] for x in batch], dim=0),
        "reg_mask": torch.stack([x["reg_mask"] for x in batch], dim=0),
        "vel_mask": torch.stack([x["vel_mask"] for x in batch], dim=0),
        "gt_boxes": [x["gt_boxes"] for x in batch],
    }

