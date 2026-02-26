import argparse
import json
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse


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


def read_split(split: str) -> List[str]:
    split_file = SPLIT_DIR / f"{split}.txt"
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_sample_id(sample_id: str) -> str:
    if not ID_RE.match(sample_id):
        raise ValueError(f"Invalid sample id: {sample_id}")
    return sample_id


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


def collect_class_hist(ids: List[str]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for sample_id in ids:
        for item in read_labels(sample_id):
            name = item["name"]
            hist[name] = hist.get(name, 0) + 1
    return dict(sorted(hist.items(), key=lambda x: (-x[1], x[0])))


def dataset_summary() -> Dict:
    train_ids = read_split("train")
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
            "split_val": len(val_ids),
            "split_train_val": len(train_val_ids),
        },
        "class_hist_train_val": collect_class_hist(train_val_ids),
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
            if split not in {"train", "val", "train_val"}:
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

