from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def lidar_box_to_kitti_fields(box_lidar: np.ndarray):
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box_lidar[:7]]

    h = dz
    w = dy
    l = dx

    z_bottom_center = z - dz * 0.5
    loc_x = -y
    loc_y = -z_bottom_center
    loc_z = x

    ry = -yaw - math.pi / 2.0
    alpha = -math.atan2(-y, x) + ry

    bbox = [0.0, 0.0, 50.0, 50.0]
    return alpha, bbox, (h, w, l), (loc_x, loc_y, loc_z), ry


def export_kitti_like_predictions(det_annos: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for anno in det_annos:
        frame_id = str(anno.get("frame_id", ""))
        names = anno.get("name", np.zeros((0,), dtype=object))
        scores = anno.get("score", np.zeros((0,), dtype=np.float32))
        boxes = anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))

        lines = []
        for name, score, box in zip(names, scores, boxes):
            alpha, bbox, dims_hwl, loc, ry = lidar_box_to_kitti_fields(box)
            h, w, l = dims_hwl
            x, y, z = loc
            lines.append(
                f"{name} -1 -1 {alpha:.6f} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} "
                f"{h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f} {float(score):.6f}"
            )

        out_file = output_dir / f"{frame_id}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
