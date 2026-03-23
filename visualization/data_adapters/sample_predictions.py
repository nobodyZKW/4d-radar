from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")


class SamplePredictionProvider:
    def __init__(self, data_root: Path, artifact_root: Path, experiments: Dict[str, Dict[str, Any]]):
        self.data_root = data_root
        self.artifact_root = artifact_root
        self.experiments = experiments

        self.image_dir = self.data_root / "lidar" / "training" / "image_2"
        self.label_dir = self.data_root / "lidar" / "training" / "label_2"
        self.radar_dir = self.data_root / "radar_5frames" / "training" / "velodyne"
        self.calib_dir = self.data_root / "lidar" / "training" / "calib"
        self.split_dir = self.data_root / "lidar" / "ImageSets"

        self._pkl_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def safe_sample_id(self, sample_id: str) -> str:
        sample_id = sample_id.strip()
        if not ID_RE.match(sample_id):
            raise ValueError(f"Invalid sample id: {sample_id}")
        return sample_id

    def list_samples(self, split: str, limit: int = 300, offset: int = 0) -> Dict[str, Any]:
        split_file = self.split_dir / f"{split}.txt"
        if not split_file.exists():
            return {"split": split, "total": 0, "offset": offset, "limit": limit, "ids": []}

        ids = [x.strip() for x in split_file.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        sliced = ids[offset : offset + max(limit, 0)]
        return {
            "split": split,
            "total": len(ids),
            "offset": offset,
            "limit": limit,
            "ids": sliced,
        }

    def _read_labels(self, sample_id: str) -> List[Dict[str, Any]]:
        label_path = self.label_dir / f"{sample_id}.txt"
        if not label_path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 15:
                continue
            rows.append(
                {
                    "name": p[0],
                    "bbox_2d": [float(p[4]), float(p[5]), float(p[6]), float(p[7])],
                    "hwl": [float(p[8]), float(p[9]), float(p[10])],
                    "loc_cam": [float(p[11]), float(p[12]), float(p[13])],
                    "ry": float(p[14]),
                }
            )
        return rows

    def _read_tr_velo_to_cam(self, sample_id: str) -> np.ndarray:
        calib_path = self.calib_dir / f"{sample_id}.txt"
        if not calib_path.exists():
            return np.eye(4, dtype=np.float32)
        for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Tr_velo_to_cam:"):
                values = [float(x) for x in line.split(":", 1)[1].strip().split()]
                if len(values) == 12:
                    tr = np.array(values, dtype=np.float32).reshape(3, 4)
                    t = np.eye(4, dtype=np.float32)
                    t[:3, :4] = tr
                    return t
        return np.eye(4, dtype=np.float32)

    def _labels_to_lidar_boxes(self, sample_id: str, labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not labels:
            return []
        tr = self._read_tr_velo_to_cam(sample_id)
        tr_inv = np.linalg.inv(tr.astype(np.float64))

        boxes: List[Dict[str, Any]] = []
        for row in labels:
            h, w, l = row["hwl"]
            x_cam, y_cam, z_cam = row["loc_cam"]
            ry = row["ry"]
            p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
            p_lidar = tr_inv @ p_cam
            yaw = -(ry + np.pi / 2.0)
            z_center = float(p_lidar[2] + h / 2.0)
            boxes.append(
                {
                    "label": row["name"],
                    "score": 1.0,
                    "bbox_2d": row["bbox_2d"],
                    "box_lidar": [
                        float(p_lidar[0]),
                        float(p_lidar[1]),
                        z_center,
                        float(l),
                        float(w),
                        float(h),
                        float(yaw),
                    ],
                }
            )
        return boxes

    def _load_radar_points(self, sample_id: str, max_points: int = 8000) -> List[List[float]]:
        bin_path = self.radar_dir / f"{sample_id}.bin"
        if not bin_path.exists():
            return []
        pts = np.fromfile(bin_path, dtype=np.float32)
        if pts.size == 0 or pts.size % 7 != 0:
            return []
        pts = pts.reshape(-1, 7)
        if pts.shape[0] > max_points:
            stride = max(1, pts.shape[0] // max_points)
            pts = pts[::stride][:max_points]
        # x, y, z, vr_comp
        return pts[:, [0, 1, 2, 5]].astype(np.float32).tolist()

    def get_sample(self, sample_id: str, max_points: int = 8000) -> Dict[str, Any]:
        sid = self.safe_sample_id(sample_id)
        labels = self._read_labels(sid)
        gt_boxes = self._labels_to_lidar_boxes(sid, labels)

        return {
            "sample_id": sid,
            "image_url": f"/data/image/{sid}.jpg",
            "gt_boxes": gt_boxes,
            "label_rows": labels,
            "radar_points": self._load_radar_points(sid, max_points=max_points),
        }

    def _load_from_prediction_dir(self, exp: Dict[str, Any], sample_id: str) -> Optional[List[Dict[str, Any]]]:
        pred_text = str(exp.get("prediction_abs", "")).strip()
        if not pred_text:
            return None
        pred_dir = Path(pred_text)
        if not pred_dir.exists() or not pred_dir.is_dir():
            return None
        json_path = pred_dir / f"{sample_id}.json"
        if not json_path.exists():
            return None
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return list(payload.get("pred_boxes", []))
            if isinstance(payload, list):
                return payload
        except Exception:
            return None
        return None

    def _find_result_pkl(self, exp: Dict[str, Any]) -> Optional[Path]:
        raw = str(exp.get("official_eval_abs", "")).strip()
        if not raw:
            return None
        path = Path(raw)
        if path.is_file() and path.name.endswith(".pkl"):
            return path
        if path.is_dir():
            cands = sorted(path.rglob("result.pkl"), key=lambda p: p.stat().st_mtime)
            return cands[-1] if cands else None
        return None

    def _load_result_pkl_index(self, pkl_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        key = str(pkl_path)
        if key in self._pkl_cache:
            return self._pkl_cache[key]

        index: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with open(pkl_path, "rb") as f:
                det_annos = pickle.load(f)
            for anno in det_annos:
                sid = str(anno.get("frame_id", "")).strip()
                if not sid:
                    continue
                names = anno.get("name", [])
                scores = anno.get("score", [])
                boxes = anno.get("boxes_lidar", [])
                rows = []
                for name, score, box in zip(names, scores, boxes):
                    box_list = [float(x) for x in box.tolist()] if hasattr(box, "tolist") else [float(x) for x in box]
                    rows.append(
                        {
                            "label": str(name),
                            "score": float(score),
                            "box_lidar": box_list,
                        }
                    )
                index[sid] = rows
        except Exception:
            index = {}

        self._pkl_cache[key] = index
        return index

    def _cache_standardized_prediction(self, exp_id: str, sample_id: str, pred_boxes: List[Dict[str, Any]]) -> None:
        out_dir = self.artifact_root / "viz_samples" / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample_id}.json"
        payload = {
            "sample_id": sample_id,
            "exp_id": exp_id,
            "pred_boxes": pred_boxes,
        }
        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_predictions_for_exp(self, exp_id: str, sample_id: str) -> Dict[str, Any]:
        exp = self.experiments.get(exp_id)
        if not exp:
            return {
                "exp_id": exp_id,
                "available": False,
                "message": "experiment not found",
                "pred_boxes": [],
            }

        pred_boxes = self._load_from_prediction_dir(exp, sample_id)
        source = "prediction_dir"
        if pred_boxes is None:
            pkl_path = self._find_result_pkl(exp)
            if pkl_path:
                index = self._load_result_pkl_index(pkl_path)
                pred_boxes = index.get(sample_id, [])
                source = f"result_pkl:{pkl_path.name}"
                self._cache_standardized_prediction(exp_id, sample_id, pred_boxes)

        if pred_boxes is None:
            return {
                "exp_id": exp_id,
                "available": False,
                "message": "prediction artifact missing",
                "pred_boxes": [],
            }

        return {
            "exp_id": exp_id,
            "available": True,
            "message": "",
            "source": source,
            "pred_boxes": pred_boxes,
        }

    def get_predictions(self, sample_id: str, exp_ids: List[str]) -> Dict[str, Any]:
        sid = self.safe_sample_id(sample_id)
        preds = [self._load_predictions_for_exp(exp_id, sid) for exp_id in exp_ids]
        return {
            "sample_id": sid,
            "predictions": preds,
        }
