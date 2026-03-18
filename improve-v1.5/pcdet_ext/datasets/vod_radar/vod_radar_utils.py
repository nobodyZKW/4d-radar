from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


DEFAULT_CLASS_MAP = {
    "Car": ["Car", "truck", "vehicle_other"],
    "Pedestrian": ["Pedestrian", "human_depiction"],
    "Cyclist": [
        "Cyclist",
        "bicycle",
        "rider",
        "moped_scooter",
        "motor",
        "ride_other",
        "ride_uncertain",
        "bicycle_rack",
    ],
}


def map_class_name(raw_name: str, class_map: Optional[Dict[str, Iterable[str]]] = None) -> str:
    mapping = class_map if class_map else DEFAULT_CLASS_MAP
    for dst, src_names in mapping.items():
        if raw_name in src_names:
            return dst
    return ""


def parse_label_line(line: str) -> Dict[str, float]:
    parts = line.strip().split()
    if len(parts) < 15:
        raise ValueError(f"Invalid label line: {line}")

    return {
        "name": parts[0],
        "truncated": float(parts[1]),
        "occluded": float(parts[2]),
        "alpha": float(parts[3]),
        "bbox": np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32),
        "h": float(parts[8]),
        "w": float(parts[9]),
        "l": float(parts[10]),
        "x": float(parts[11]),
        "y": float(parts[12]),
        "z": float(parts[13]),
        "ry": float(parts[14]),
        "score": float(parts[15]) if len(parts) > 15 else 1.0,
    }


def parse_calib_matrix(calib_path: Path, key: str, shape: Tuple[int, int]) -> np.ndarray:
    prefix = f"{key}:"
    for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(prefix):
            values = [float(x) for x in line.split(":", 1)[1].strip().split()]
            mat = np.array(values, dtype=np.float32).reshape(shape)
            return mat
    raise KeyError(f"{key} not found in {calib_path}")


def load_vod_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    p2 = parse_calib_matrix(calib_path, "P2", (3, 4))
    r0 = parse_calib_matrix(calib_path, "R0_rect", (3, 3))
    v2c = parse_calib_matrix(calib_path, "Tr_velo_to_cam", (3, 4))

    p2_4x4 = np.eye(4, dtype=np.float32)
    p2_4x4[:3, :4] = p2

    r0_4x4 = np.eye(4, dtype=np.float32)
    r0_4x4[:3, :3] = r0

    v2c_4x4 = np.eye(4, dtype=np.float32)
    v2c_4x4[:3, :4] = v2c

    return {
        "P2": p2_4x4,
        "R0_rect": r0_4x4,
        "Tr_velo_to_cam": v2c_4x4,
    }


def camera_box_to_lidar(box_cam: Dict[str, float], tr_velo_to_cam: np.ndarray) -> np.ndarray:
    """Convert KITTI camera box to lidar box [x, y, z, dx, dy, dz, heading]."""
    t_inv = np.linalg.inv(tr_velo_to_cam.astype(np.float64))
    p_cam = np.array([box_cam["x"], box_cam["y"], box_cam["z"], 1.0], dtype=np.float64)
    p_lidar = t_inv @ p_cam

    h = float(box_cam["h"])
    w = float(box_cam["w"])
    l = float(box_cam["l"])
    yaw = -(float(box_cam["ry"]) + math.pi / 2.0)
    z_center = float(p_lidar[2] + h / 2.0)

    return np.array([p_lidar[0], p_lidar[1], z_center, l, w, h, yaw], dtype=np.float32)


def get_image_shape(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        w, h = img.size
    return np.array([h, w], dtype=np.int32)


def resolve_image_file(image_dir: Path, sample_id: str) -> Optional[Path]:
    for ext in (".jpg", ".png", ".jpeg", ".bmp"):
        p = image_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None


def parse_vod_label_file(
    label_path: Path,
    tr_velo_to_cam: np.ndarray,
    class_map: Optional[Dict[str, Iterable[str]]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    names, truncs, occs, alphas, bboxes = [], [], [], [], []
    dims, locs, rots, scores = [], [], [], []
    gt_boxes_lidar = []

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = parse_label_line(line)
        mapped = map_class_name(obj["name"], class_map)
        if mapped == "":
            continue
        if class_names is not None and mapped not in class_names:
            continue

        box_lidar = camera_box_to_lidar(obj, tr_velo_to_cam)
        names.append(mapped)
        truncs.append(obj["truncated"])
        occs.append(obj["occluded"])
        alphas.append(obj["alpha"])
        bboxes.append(obj["bbox"])
        dims.append(np.array([obj["l"], obj["h"], obj["w"]], dtype=np.float32))
        locs.append(np.array([obj["x"], obj["y"], obj["z"]], dtype=np.float32))
        rots.append(obj["ry"])
        scores.append(obj["score"])
        gt_boxes_lidar.append(box_lidar)

    if len(names) == 0:
        return {
            "name": np.zeros((0,), dtype=object),
            "truncated": np.zeros((0,), dtype=np.float32),
            "occluded": np.zeros((0,), dtype=np.float32),
            "alpha": np.zeros((0,), dtype=np.float32),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "location": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.zeros((0,), dtype=np.float32),
            "score": np.zeros((0,), dtype=np.float32),
            "difficulty": np.zeros((0,), dtype=np.int32),
            "index": np.zeros((0,), dtype=np.int32),
            "gt_boxes_lidar": np.zeros((0, 7), dtype=np.float32),
        }

    return {
        "name": np.array(names),
        "truncated": np.array(truncs, dtype=np.float32),
        "occluded": np.array(occs, dtype=np.float32),
        "alpha": np.array(alphas, dtype=np.float32),
        "bbox": np.stack(bboxes, axis=0).astype(np.float32),
        "dimensions": np.stack(dims, axis=0).astype(np.float32),
        "location": np.stack(locs, axis=0).astype(np.float32),
        "rotation_y": np.array(rots, dtype=np.float32),
        "score": np.array(scores, dtype=np.float32),
        "difficulty": np.zeros((len(names),), dtype=np.int32),
        "index": np.arange(len(names), dtype=np.int32),
        "gt_boxes_lidar": np.stack(gt_boxes_lidar, axis=0).astype(np.float32),
    }
