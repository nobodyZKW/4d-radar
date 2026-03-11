import argparse
import json
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import numpy as np


VIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VIS_ROOT.parent
WEB_ROOT = VIS_ROOT / "web"
DATA_ROOT = PROJECT_ROOT.parent / "vod-min"
BASELINE_ROOT = PROJECT_ROOT / "baseline"
HISTORY_PATH = BASELINE_ROOT / "outputs" / "vod_baseline" / "history.json"
SPLIT_DIR = DATA_ROOT / "lidar" / "ImageSets"
LABEL_DIR = DATA_ROOT / "lidar" / "training" / "label_2"
IMAGE_DIR = DATA_ROOT / "lidar" / "training" / "image_2"
RADAR_DIR = DATA_ROOT / "radar_5frames" / "training" / "velodyne"
CALIB_DIR = DATA_ROOT / "lidar" / "training" / "calib"

ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")
VALID_SPLITS = {"train", "val", "test", "train_val"}


def read_split(split: str) -> List[str]:
    split_file = SPLIT_DIR / f"{split}.txt"
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_sample_id(sample_id: str) -> str:
    if not ID_RE.match(sample_id):
        raise ValueError(f"Invalid sample id: {sample_id}")
    return sample_id


def parse_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(min_value, min(max_value, parsed))


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


@lru_cache(maxsize=8192)
def read_tr_velo_to_cam(sample_id: str) -> np.ndarray:
    sample_id = safe_sample_id(sample_id)
    calib_path = CALIB_DIR / f"{sample_id}.txt"
    if not calib_path.exists():
        return np.eye(4, dtype=np.float32)
    for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Tr_velo_to_cam:"):
            values = [float(x) for x in line.split(":", 1)[1].strip().split()]
            if len(values) != 12:
                break
            tr = np.array(values, dtype=np.float32).reshape(3, 4)
            t = np.eye(4, dtype=np.float32)
            t[:3, :4] = tr
            return t
    return np.eye(4, dtype=np.float32)


@lru_cache(maxsize=8192)
def read_lidar_boxes(sample_id: str) -> List[Dict]:
    sample_id = safe_sample_id(sample_id)
    label_path = LABEL_DIR / f"{sample_id}.txt"
    if not label_path.exists():
        return []
    tr = read_tr_velo_to_cam(sample_id)
    tr_inv = np.linalg.inv(tr.astype(np.float64))
    boxes: List[Dict] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        p = line.split()
        if len(p) < 15:
            continue
        try:
            name = p[0]
            h = float(p[8])
            w = float(p[9])
            l = float(p[10])
            x_cam = float(p[11])
            y_cam = float(p[12])
            z_cam = float(p[13])
            ry = float(p[14])
        except Exception:
            continue

        p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
        p_lidar = tr_inv @ p_cam
        yaw = -(ry + np.pi / 2.0)
        z_center = float(p_lidar[2] + h / 2.0)
        boxes.append(
            {
                "name": name,
                "cx": float(p_lidar[0]),
                "cy": float(p_lidar[1]),
                "cz": z_center,
                "l": float(l),
                "w": float(w),
                "h": float(h),
                "yaw": float(yaw),
            }
        )
    return boxes


@lru_cache(maxsize=8192)
def read_labels(sample_id: str) -> List[Dict]:
    sample_id = safe_sample_id(sample_id)
    label_path = LABEL_DIR / f"{sample_id}.txt"
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        p = line.split()
        if len(p) < 15:
            continue
        rows.append(
            {
                "name": p[0],
                "truncated": float(p[1]),
                "occluded": int(float(p[2])),
                "alpha": float(p[3]),
                "bbox": [float(p[4]), float(p[5]), float(p[6]), float(p[7])],
            }
        )
    return rows


@lru_cache(maxsize=8192)
def read_radar_points(sample_id: str) -> np.ndarray:
    sample_id = safe_sample_id(sample_id)
    radar_path = RADAR_DIR / f"{sample_id}.bin"
    if not radar_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    pts = np.fromfile(radar_path, dtype=np.float32)
    if pts.size == 0 or pts.size % 7 != 0:
        return np.zeros((0, 4), dtype=np.float32)
    pts = pts.reshape(-1, 7)
    return pts[:, [0, 1, 2, 4]].astype(np.float32)


def label_points_by_boxes(points: np.ndarray, boxes: List[Dict]) -> Dict:
    n = int(points.shape[0])
    if n == 0:
        return {"label_ids": np.zeros((0,), dtype=np.int16), "class_names": ["background"]}
    class_names: List[str] = ["background"]
    name_to_id: Dict[str, int] = {}
    label_ids = np.zeros((n,), dtype=np.int16)
    unlabeled = np.ones((n,), dtype=bool)

    for box in boxes:
        if not np.any(unlabeled):
            break
        name = box["name"]
        if name not in name_to_id:
            name_to_id[name] = len(class_names)
            class_names.append(name)
        cls_id = name_to_id[name]

        idx = np.where(unlabeled)[0]
        pts = points[idx]
        dx = pts[:, 0] - box["cx"]
        dy = pts[:, 1] - box["cy"]
        dz = pts[:, 2] - box["cz"]
        cos_y = np.cos(box["yaw"])
        sin_y = np.sin(box["yaw"])
        local_x = cos_y * dx + sin_y * dy
        local_y = -sin_y * dx + cos_y * dy

        inside = (
            (np.abs(local_x) <= box["l"] * 0.5)
            & (np.abs(local_y) <= box["w"] * 0.5)
            & (np.abs(dz) <= box["h"] * 0.5)
        )
        if np.any(inside):
            hit_idx = idx[inside]
            label_ids[hit_idx] = cls_id
            unlabeled[hit_idx] = False

    return {"label_ids": label_ids, "class_names": class_names}


def collect_class_hist(ids: List[str]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for sample_id in ids:
        for item in read_labels(sample_id):
            name = item["name"]
            hist[name] = hist.get(name, 0) + 1
    return dict(sorted(hist.items(), key=lambda x: (-x[1], x[0])))


@lru_cache(maxsize=1)
def dataset_summary() -> Dict:
    train_ids = read_split("train")
    test_ids = read_split("test")
    val_ids = read_split("val")
    train_val_ids = read_split("train_val")
    return {
        "paths": {
            "data_root": str(DATA_ROOT),
            "baseline_root": str(BASELINE_ROOT),
        },
        "counts": {
            "images": len(list(IMAGE_DIR.glob("*.jpg"))) if IMAGE_DIR.exists() else 0,
            "radar_bins": len(list(RADAR_DIR.glob("*.bin"))) if RADAR_DIR.exists() else 0,
            "labels": len(list(LABEL_DIR.glob("*.txt"))) if LABEL_DIR.exists() else 0,
            "calib": len(list(CALIB_DIR.glob("*.txt"))) if CALIB_DIR.exists() else 0,
            "split_train": len(train_ids),
            "split_test": len(test_ids),
            "split_val": len(val_ids),
            "split_train_val": len(train_val_ids),
        },
        "class_hist": {
            "train": collect_class_hist(train_ids),
            "test": collect_class_hist(test_ids),
            "val": collect_class_hist(val_ids),
            "train_val": collect_class_hist(train_val_ids),
        },
    }


def read_history() -> Dict:
    if not HISTORY_PATH.exists():
        return {
            "exists": False,
            "history": [],
            "latest": {},
            "best_mean_f1": {},
            "path": str(HISTORY_PATH),
        }
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    latest = history[-1] if history else {}
    best = {}
    if history:
        best = max(history, key=lambda x: float(x.get("mean_f1", -1.0)))
    return {
        "exists": True,
        "history": history,
        "latest": latest,
        "best_mean_f1": best,
        "path": str(HISTORY_PATH),
    }


class RadarVisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _send_json(self, payload: Dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json({"ok": True})
            return

        if path == "/api/summary":
            self._send_json(dataset_summary())
            return

        if path == "/api/history":
            self._send_json(read_history())
            return

        if path == "/api/samples":
            split = query.get("split", ["val"])[0]
            if split not in VALID_SPLITS:
                self._send_json({"error": "invalid split"}, code=400)
                return
            limit = int(query.get("limit", ["300"])[0])
            offset = int(query.get("offset", ["0"])[0])
            ids = read_split(split)
            sliced = ids[offset : offset + max(limit, 0)]
            self._send_json({"split": split, "offset": offset, "limit": limit, "total": len(ids), "ids": sliced})
            return

        if path.startswith("/api/labels/"):
            sample_id = path.split("/api/labels/", 1)[1]
            try:
                safe_id = safe_sample_id(sample_id)
            except ValueError as e:
                self._send_json({"error": str(e)}, code=400)
                return
            self._send_json({"id": safe_id, "labels": read_labels(safe_id)})
            return

        if path.startswith("/api/radar/"):
            sample_id = path.split("/api/radar/", 1)[1]
            try:
                safe_id = safe_sample_id(sample_id)
            except ValueError as e:
                self._send_json({"error": str(e)}, code=400)
                return
            max_points = parse_int(query.get("max_points", ["12000"])[0], default=12000, min_value=100, max_value=50000)
            x_min = parse_float(query.get("x_min", ["0"])[0], 0.0)
            x_max = parse_float(query.get("x_max", ["60"])[0], 60.0)
            y_min = parse_float(query.get("y_min", ["-30"])[0], -30.0)
            y_max = parse_float(query.get("y_max", ["30"])[0], 30.0)
            color_by = query.get("color_by", ["velocity"])[0].strip().lower()
            if color_by not in {"velocity", "label"}:
                color_by = "velocity"
            points = read_radar_points(safe_id)
            if points.shape[0] > 0:
                mask = (
                    (points[:, 0] >= x_min)
                    & (points[:, 0] <= x_max)
                    & (points[:, 1] >= y_min)
                    & (points[:, 1] <= y_max)
                )
                points = points[mask]
            total = int(points.shape[0])
            if max_points > 0 and points.shape[0] > max_points:
                stride = max(1, points.shape[0] // max_points)
                points = points[::stride][:max_points]

            payload: Dict = {
                "id": safe_id,
                "mode": color_by,
                "total_points": total,
                "returned_points": int(points.shape[0]),
                "range": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            }

            if color_by == "label":
                labeled = label_points_by_boxes(points, read_lidar_boxes(safe_id))
                label_ids = labeled["label_ids"]
                class_names = labeled["class_names"]
                if points.shape[0] > 0:
                    out_points = np.column_stack((points[:, 0], points[:, 1], points[:, 3], label_ids))
                else:
                    out_points = np.zeros((0, 4), dtype=np.float32)
                uniq, cnt = np.unique(label_ids, return_counts=True) if label_ids.size > 0 else ([], [])
                label_hist: Dict[str, int] = {}
                for lid, c in zip(uniq, cnt):
                    i = int(lid)
                    name = class_names[i] if 0 <= i < len(class_names) else f"class_{i}"
                    label_hist[name] = int(c)
                payload["class_names"] = class_names
                payload["label_hist"] = label_hist
                payload["points"] = out_points.tolist()
            else:
                payload["points"] = points[:, [0, 1, 3]].tolist() if points.shape[0] > 0 else []

            self._send_json(payload)
            return

        if path.startswith("/data/image/"):
            sample_name = path.split("/data/image/", 1)[1]
            if not sample_name.endswith(".jpg"):
                self.send_error(HTTPStatus.BAD_REQUEST, "Only .jpg is supported")
                return
            sample_id = sample_name[:-4]
            try:
                safe_id = safe_sample_id(sample_id)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid sample id")
                return
            self._send_file(IMAGE_DIR / f"{safe_id}.jpg", "image/jpeg")
            return

        if path == "/":
            self.path = "/index.html"
        if path == "/split-player":
            self.path = "/split_player.html"
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar baseline visualization server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Web folder not found: {WEB_ROOT}")

    print(f"[visualization] serving web: {WEB_ROOT}")
    print(f"[visualization] data root: {DATA_ROOT}")
    print(f"[visualization] open: http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), RadarVisHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
