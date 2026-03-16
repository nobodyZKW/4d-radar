import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def read_ids(split_file: Path) -> List[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_calib_tr_velo_to_cam(calib_path: Path) -> np.ndarray:
    for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Tr_velo_to_cam:"):
            values = [float(x) for x in line.split(":", 1)[1].strip().split()]
            tr = np.array(values, dtype=np.float32).reshape(3, 4)
            t = np.eye(4, dtype=np.float32)
            t[:3, :4] = tr
            return t
    raise ValueError(f"Tr_velo_to_cam not found in {calib_path}")


def parse_label_line(line: str) -> Dict[str, float]:
    p = line.strip().split()
    if len(p) < 15:
        raise ValueError(f"Invalid label line: {line}")
    return {
        "name": p[0],
        "h": float(p[8]),
        "w": float(p[9]),
        "l": float(p[10]),
        "x": float(p[11]),
        "y": float(p[12]),
        "z": float(p[13]),
        "ry": float(p[14]),
    }


def camera_box_to_lidar(box_cam: Dict[str, float], tr_velo_to_cam: np.ndarray) -> np.ndarray:
    """KITTI camera box -> lidar box [x, y, z, l, w, h, yaw]."""
    t_inv = np.linalg.inv(tr_velo_to_cam.astype(np.float64))
    p_cam = np.array([box_cam["x"], box_cam["y"], box_cam["z"], 1.0], dtype=np.float64)
    p_lidar = t_inv @ p_cam

    h = float(box_cam["h"])
    w = float(box_cam["w"])
    l = float(box_cam["l"])
    yaw = -(float(box_cam["ry"]) + math.pi / 2.0)
    z_center = float(p_lidar[2] + h / 2.0)

    return np.array([p_lidar[0], p_lidar[1], z_center, l, w, h, yaw], dtype=np.float32)


def map_class(raw_name: str, class_map: Dict[str, Iterable[str]]) -> str:
    for dst, src_names in class_map.items():
        if raw_name in src_names:
            return dst
    return ""


def gaussian2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    y, x = center
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=max(1.0, diameter / 6.0))

    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    masked_gaussian = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    def state_dict(self) -> Dict:
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.ema.load_state_dict(state_dict, strict=True)

