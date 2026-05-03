from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class VelocityEstimate:
    velocity_xy: np.ndarray
    valid: bool
    weak: bool
    quality: str
    residual: float
    num_points: int
    condition_number: float
    weight: float


class VelocitySupervisionEstimator:
    """Estimate per-box BEV velocity from compensated radial Doppler constraints."""

    def __init__(self, cfg: Dict):
        self.min_points = int(cfg.get("MIN_POINTS", 3))
        self.max_speed = float(cfg.get("MAX_SPEED", 35.0))
        self.max_residual = float(cfg.get("MAX_RESIDUAL", 2.5))
        self.max_condition = float(cfg.get("MAX_CONDITION", 5000.0))
        self.huber_delta = float(cfg.get("HUBER_DELTA", 1.5))
        self.irls_iters = int(cfg.get("IRLS_ITERS", 5))
        self.reg_lambda = float(cfg.get("REG_LAMBDA", 1e-3))
        self.weak_weight = float(cfg.get("WEAK_WEIGHT", 0.35))
        self.strong_weight = float(cfg.get("STRONG_WEIGHT", 1.0))
        self.use_fallback_heading = bool(cfg.get("USE_FALLBACK_HEADING", True))
        supervision_mode = str(cfg.get("SUPERVISION_MODE", "robust")).strip().lower()
        if supervision_mode in {"none", "disabled", "off"}:
            self.supervision_mode = "disabled"
        elif supervision_mode in {"weak", "weak_only", "heading", "fallback"}:
            self.supervision_mode = "weak_only"
        else:
            self.supervision_mode = "robust"

    @staticmethod
    def _points_in_rotated_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
        if points_xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        cx, cy, cz, dx, dy, dz, yaw = [float(x) for x in box[:7]]
        rel_x = points_xyz[:, 0] - cx
        rel_y = points_xyz[:, 1] - cy
        rel_z = points_xyz[:, 2] - cz

        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        local_x = c * rel_x + s * rel_y
        local_y = -s * rel_x + c * rel_y

        return (np.abs(local_x) <= dx * 0.5) & (np.abs(local_y) <= dy * 0.5) & (np.abs(rel_z) <= dz * 0.5)

    @staticmethod
    def _safe_normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        v_min = float(np.percentile(values, 10))
        v_max = float(np.percentile(values, 90))
        denom = max(v_max - v_min, 1e-3)
        return np.clip((values - v_min) / denom, 0.0, 1.0)

    def _build_base_weights(
        self,
        box: np.ndarray,
        pts_xyz: np.ndarray,
        rcs: np.ndarray,
        time_vals: np.ndarray,
    ) -> np.ndarray:
        if pts_xyz.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        center = box[:2].reshape(1, 2)
        dist = np.linalg.norm(pts_xyz[:, :2] - center, axis=1)
        dist_scale = max(float(max(box[3], box[4])), 1e-3)
        dist_w = np.exp(-np.square(dist / dist_scale))

        rcs_norm = self._safe_normalize(rcs)
        rcs_w = 0.5 + 0.5 * rcs_norm

        if time_vals.size > 0:
            t = time_vals - np.min(time_vals)
            tau = max(float(np.std(t)), 1e-2)
            time_w = np.exp(-t / tau)
        else:
            time_w = np.ones_like(dist_w)

        w = dist_w * rcs_w * time_w
        return np.clip(w.astype(np.float32), 1e-4, None)

    def _weighted_lstsq(self, u: np.ndarray, b: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
        w = np.clip(w, 1e-6, None)
        uw = u * w[:, None]
        ata = uw.T @ u + np.eye(2, dtype=np.float64) * self.reg_lambda
        atb = uw.T @ b

        cond = float(np.linalg.cond(ata))
        if not np.isfinite(cond):
            cond = float("inf")

        try:
            v = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            v = np.array([np.nan, np.nan], dtype=np.float64)
            cond = float("inf")
        return v.astype(np.float32), cond

    def _robust_velocity_fit(
        self,
        box: np.ndarray,
        pts_xyz: np.ndarray,
        rcs: np.ndarray,
        vr_comp: np.ndarray,
        time_vals: np.ndarray,
    ) -> VelocityEstimate:
        n = int(pts_xyz.shape[0])
        invalid = VelocityEstimate(
            velocity_xy=np.array([np.nan, np.nan], dtype=np.float32),
            valid=False,
            weak=False,
            quality="invalid",
            residual=float("nan"),
            num_points=n,
            condition_number=float("inf"),
            weight=0.0,
        )
        if n < self.min_points:
            return invalid

        xy = pts_xyz[:, :2]
        rr = np.linalg.norm(xy, axis=1)
        valid_range = rr > 1e-3
        if int(valid_range.sum()) < self.min_points:
            return invalid

        xy = xy[valid_range]
        rr = rr[valid_range]
        b = vr_comp[valid_range]
        rcs = rcs[valid_range]
        time_vals = time_vals[valid_range]

        u = xy / rr[:, None]
        w_base = self._build_base_weights(box, pts_xyz[valid_range], rcs, time_vals).astype(np.float64)

        v, cond = self._weighted_lstsq(u.astype(np.float64), b.astype(np.float64), w_base)
        if not np.isfinite(v).all() or cond > self.max_condition:
            return invalid

        for _ in range(self.irls_iters):
            residual = b - (u @ v)
            mad = np.median(np.abs(residual - np.median(residual)))
            scale = max(1.4826 * mad, 1e-2)
            cutoff = self.huber_delta * scale
            huber = np.ones_like(residual, dtype=np.float64)
            over = np.abs(residual) > cutoff
            huber[over] = cutoff / (np.abs(residual[over]) + 1e-6)
            w_iter = w_base * huber
            v_new, cond = self._weighted_lstsq(u.astype(np.float64), b.astype(np.float64), w_iter)
            if not np.isfinite(v_new).all() or cond > self.max_condition:
                return invalid
            v = v_new

        residual = b - (u @ v)
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        speed = float(np.linalg.norm(v))

        if (not np.isfinite(speed)) or speed > self.max_speed or rmse > self.max_residual:
            return invalid

        return VelocityEstimate(
            velocity_xy=v.astype(np.float32),
            valid=True,
            weak=False,
            quality="strong",
            residual=rmse,
            num_points=int(valid_range.sum()),
            condition_number=cond,
            weight=self.strong_weight,
        )

    def _fallback_heading_projection(self, box: np.ndarray, vr_comp: np.ndarray) -> VelocityEstimate:
        if vr_comp.size == 0:
            return VelocityEstimate(
                velocity_xy=np.array([np.nan, np.nan], dtype=np.float32),
                valid=False,
                weak=False,
                quality="invalid",
                residual=float("nan"),
                num_points=0,
                condition_number=float("inf"),
                weight=0.0,
            )

        comp = float(np.median(vr_comp))
        yaw = float(box[6])
        vx = comp * float(np.cos(yaw))
        vy = comp * float(np.sin(yaw))
        speed = float(np.hypot(vx, vy))
        if speed > self.max_speed:
            return VelocityEstimate(
                velocity_xy=np.array([np.nan, np.nan], dtype=np.float32),
                valid=False,
                weak=False,
                quality="invalid",
                residual=float("nan"),
                num_points=int(vr_comp.size),
                condition_number=float("inf"),
                weight=0.0,
            )

        return VelocityEstimate(
            velocity_xy=np.array([vx, vy], dtype=np.float32),
            valid=True,
            weak=True,
            quality="weak",
            residual=float("nan"),
            num_points=int(vr_comp.size),
            condition_number=float("inf"),
            weight=self.weak_weight,
        )

    def estimate_for_boxes(
        self,
        points: np.ndarray,
        gt_boxes_lidar: np.ndarray,
        feature_indices: Dict[str, int],
    ) -> Tuple[np.ndarray, List[VelocityEstimate], Dict[str, float]]:
        """
        Returns:
            vel_targets: (N, 3) -> [vx, vy, vel_weight]
            estimates: list of VelocityEstimate
            stats: aggregate scalar stats for logging
        """
        num_boxes = int(gt_boxes_lidar.shape[0])
        vel_targets = np.full((num_boxes, 3), np.nan, dtype=np.float32)
        if num_boxes == 0:
            return vel_targets, [], {
                "num_valid_vel_boxes": 0.0,
                "num_weak_vel_boxes": 0.0,
                "num_strong_vel_boxes": 0.0,
                "num_invalid_vel_boxes": 0.0,
                "num_total_gt_boxes": 0.0,
                "vel_fit_residual_mean": 0.0,
                "velocity_branch_activation_ratio": 0.0,
                "velocity_weak_ratio": 0.0,
                "velocity_strong_ratio": 0.0,
            }

        xyz = points[:, :3]
        rcs = points[:, int(feature_indices.get("rcs", 3))] if points.shape[1] > int(feature_indices.get("rcs", 3)) else np.zeros((points.shape[0],), dtype=np.float32)
        vr_comp = points[:, int(feature_indices.get("v_r_comp", 5))] if points.shape[1] > int(feature_indices.get("v_r_comp", 5)) else np.zeros((points.shape[0],), dtype=np.float32)
        time_vals = points[:, int(feature_indices.get("time", 6))] if points.shape[1] > int(feature_indices.get("time", 6)) else np.zeros((points.shape[0],), dtype=np.float32)

        estimates: List[VelocityEstimate] = []
        residuals = []
        num_valid = 0
        num_weak = 0

        for bi in range(num_boxes):
            box = gt_boxes_lidar[bi]
            mask = self._points_in_rotated_box(xyz, box)
            pts_xyz = xyz[mask]
            pts_rcs = rcs[mask]
            pts_vr_comp = vr_comp[mask]
            pts_time = time_vals[mask]

            if self.supervision_mode == "disabled":
                est = VelocityEstimate(
                    velocity_xy=np.array([np.nan, np.nan], dtype=np.float32),
                    valid=False,
                    weak=False,
                    quality="disabled",
                    residual=float("nan"),
                    num_points=int(pts_xyz.shape[0]),
                    condition_number=float("inf"),
                    weight=0.0,
                )
            elif self.supervision_mode == "weak_only":
                est = self._fallback_heading_projection(box, pts_vr_comp)
            else:
                est = self._robust_velocity_fit(
                    box=box,
                    pts_xyz=pts_xyz,
                    rcs=pts_rcs,
                    vr_comp=pts_vr_comp,
                    time_vals=pts_time,
                )

                if (not est.valid) and self.use_fallback_heading:
                    est = self._fallback_heading_projection(box, pts_vr_comp)

            estimates.append(est)

            if est.valid:
                vel_targets[bi, 0:2] = est.velocity_xy
                vel_targets[bi, 2] = float(est.weight)
                num_valid += 1
                if est.weak:
                    num_weak += 1
                if np.isfinite(est.residual):
                    residuals.append(float(est.residual))
            else:
                vel_targets[bi, 2] = 0.0

        num_strong = max(num_valid - num_weak, 0)
        num_invalid = max(num_boxes - num_valid, 0)
        stats = {
            "num_valid_vel_boxes": float(num_valid),
            "num_weak_vel_boxes": float(num_weak),
            "num_strong_vel_boxes": float(num_strong),
            "num_invalid_vel_boxes": float(num_invalid),
            "num_total_gt_boxes": float(num_boxes),
            "vel_fit_residual_mean": float(np.mean(residuals)) if residuals else 0.0,
            "velocity_branch_activation_ratio": float(num_valid / max(num_boxes, 1)),
            "velocity_weak_ratio": float(num_weak / max(num_valid, 1)),
            "velocity_strong_ratio": float(num_strong / max(num_valid, 1)),
        }
        return vel_targets, estimates, stats
