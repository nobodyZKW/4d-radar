import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_history(history_path: Path) -> List[Dict]:
    if not history_path.exists():
        raise FileNotFoundError(f"history not found: {history_path}")
    return json.loads(history_path.read_text(encoding="utf-8"))


def plot_loss_curve(history: List[Dict], out_path: Path) -> None:
    epochs = [int(x["epoch"]) for x in history]
    train_loss = [float(x.get("loss", 0.0)) for x in history]
    val_loss = [float(x.get("val_loss", 0.0)) for x in history]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Loss Curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_f1_curve(history: List[Dict], out_path: Path) -> None:
    epochs = [int(x["epoch"]) for x in history]
    mean_f1 = [float(x.get("mean_f1", 0.0)) for x in history]
    car_f1 = [float(x.get("Car_f1", 0.0)) for x in history]
    ped_f1 = [float(x.get("Pedestrian_f1", 0.0)) for x in history]
    cyc_f1 = [float(x.get("Cyclist_f1", 0.0)) for x in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, mean_f1, label="mean_f1", linewidth=2.4)
    plt.plot(epochs, car_f1, label="Car_f1", linewidth=1.8)
    plt.plot(epochs, ped_f1, label="Pedestrian_f1", linewidth=1.8)
    plt.plot(epochs, cyc_f1, label="Cyclist_f1", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Baseline F1 Curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_latest_class_metrics(history: List[Dict], out_path: Path) -> Dict[str, float]:
    latest = history[-1]
    classes = ["Car", "Pedestrian", "Cyclist"]
    precision = [float(latest.get(f"{c}_precision", 0.0)) for c in classes]
    recall = [float(latest.get(f"{c}_recall", 0.0)) for c in classes]
    f1 = [float(latest.get(f"{c}_f1", 0.0)) for c in classes]

    x = np.arange(len(classes))
    w = 0.24
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, precision, width=w, label="Precision")
    plt.bar(x, recall, width=w, label="Recall")
    plt.bar(x + w, f1, width=w, label="F1")
    plt.xticks(x, classes)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Latest Class Metrics (epoch={latest.get('epoch')}, mean_f1={latest.get('mean_f1', 0):.4f})")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()

    metrics = {"epoch": int(latest.get("epoch", -1)), "mean_f1": float(latest.get("mean_f1", 0.0))}
    for c in classes:
        metrics[f"{c}_precision"] = float(latest.get(f"{c}_precision", 0.0))
        metrics[f"{c}_recall"] = float(latest.get(f"{c}_recall", 0.0))
        metrics[f"{c}_f1"] = float(latest.get(f"{c}_f1", 0.0))
    return metrics


def parse_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        name = parts[0]
        x1, y1, x2, y2 = map(float, parts[4:8])
        boxes.append((name, x1, y1, x2, y2))
    return boxes


def load_radar_points(bin_path: Path, point_dim: int = 7) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32)
    if pts.size % point_dim != 0:
        raise ValueError(f"Invalid point shape in {bin_path}, floats={pts.size}, point_dim={point_dim}")
    return pts.reshape(-1, point_dim)


def pick_sample_id(data_root: Path, preferred_id: str) -> str:
    image_dir = data_root / "lidar" / "training" / "image_2"
    radar_dir = data_root / "radar_5frames" / "training" / "velodyne"
    if (image_dir / f"{preferred_id}.jpg").exists() and (radar_dir / f"{preferred_id}.bin").exists():
        return preferred_id

    split_dir = data_root / "lidar" / "ImageSets"
    for split_name in ["test.txt", "val.txt", "train.txt", "train_val.txt"]:
        split_file = split_dir / split_name
        if not split_file.exists():
            continue
        ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        for sid in ids:
            if (image_dir / f"{sid}.jpg").exists() and (radar_dir / f"{sid}.bin").exists():
                return sid

    jpgs = sorted(image_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"no image found under {image_dir}")
    return jpgs[0].stem


def render_sample_vis(data_root: Path, sample_id: str, out_path: Path) -> None:
    image_path = data_root / "lidar" / "training" / "image_2" / f"{sample_id}.jpg"
    label_path = data_root / "lidar" / "training" / "label_2" / f"{sample_id}.txt"
    radar_path = data_root / "radar_5frames" / "training" / "velodyne" / f"{sample_id}.bin"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not radar_path.exists():
        raise FileNotFoundError(f"Radar bin not found: {radar_path}")

    image = np.array(Image.open(image_path).convert("RGB"))
    boxes = parse_labels(label_path)
    pts = load_radar_points(radar_path, point_dim=7)
    mask = (pts[:, 0] >= 0.0) & (pts[:, 0] <= 60.0) & (pts[:, 1] >= -30.0) & (pts[:, 1] <= 30.0)
    pts = pts[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title(f"Image + 2D Labels ({sample_id})")
    axes[0].axis("off")
    for name, x1, y1, x2, y2 in boxes:
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        rect = plt.Rectangle((x1, y1), w, h, fill=False, linewidth=1.2)
        axes[0].add_patch(rect)
        axes[0].text(x1, max(0.0, y1 - 3.0), name, fontsize=8, color="yellow", backgroundcolor="black")

    if pts.shape[0] > 0:
        sc = axes[1].scatter(pts[:, 0], pts[:, 1], c=pts[:, 4], s=2.0, cmap="coolwarm", alpha=0.8)
        cbar = plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Radial velocity")
    axes[1].set_title(f"Radar BEV ({sample_id})")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_xlim(0.0, 60.0)
    axes[1].set_ylim(-30.0, 30.0)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser("Generate README result figures for baseline")
    parser.add_argument("--data-root", type=str, default="../vod-min")
    parser.add_argument("--history", type=str, default="baseline/outputs/vod_baseline/history.json")
    parser.add_argument("--out-dir", type=str, default="baseline/results")
    parser.add_argument("--sample-id", type=str, default="00000")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    history_path = resolve_path(args.history)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    if not history:
        raise RuntimeError(f"Empty history: {history_path}")

    loss_path = out_dir / "loss_curve.png"
    f1_path = out_dir / "f1_curve.png"
    class_path = out_dir / "latest_class_metrics.png"
    sample_path = out_dir / "sample_vis_00000.png"

    plot_loss_curve(history, loss_path)
    plot_f1_curve(history, f1_path)
    latest_metrics = plot_latest_class_metrics(history, class_path)
    (out_dir / "latest_metrics.json").write_text(json.dumps(latest_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_id = pick_sample_id(data_root, args.sample_id)
    render_sample_vis(data_root, selected_id, sample_path)

    print(f"saved: {loss_path}")
    print(f"saved: {f1_path}")
    print(f"saved: {class_path}")
    print(f"saved: {sample_path} (source id={selected_id})")
    print(f"saved: {out_dir / 'latest_metrics.json'}")


if __name__ == "__main__":
    main()

