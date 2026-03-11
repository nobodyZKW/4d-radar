import argparse
from pathlib import Path
from typing import List, Tuple

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


def parse_labels(label_path: Path) -> List[Tuple[str, float, float, float, float]]:
    boxes: List[Tuple[str, float, float, float, float]] = []
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


def draw_one_sample(
    data_root: Path,
    sample_id: str,
    save_path: Path,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> None:
    image_path = data_root / "lidar" / "training" / "image_2" / f"{sample_id}.jpg"
    label_path = data_root / "lidar" / "training" / "label_2" / f"{sample_id}.txt"
    radar_path = data_root / "radar_5frames" / "training" / "velodyne" / f"{sample_id}.bin"

    if not image_path.exists() or not radar_path.exists():
        return

    image = np.array(Image.open(image_path).convert("RGB"))
    boxes = parse_labels(label_path)
    pts = load_radar_points(radar_path, point_dim=7)

    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
    )
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
        sc = axes[1].scatter(
            pts[:, 0],
            pts[:, 1],
            c=pts[:, 4],
            s=2.0,
            cmap="coolwarm",
            alpha=0.8,
        )
        cbar = plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Radial velocity")
    axes[1].set_title(f"Radar BEV ({sample_id})")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser("Generate split visualization results (left image, right BEV)")
    parser.add_argument("--data-root", type=str, default="vod-min")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "train_val"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="baseline/results/test_vis")
    parser.add_argument("--x-range", type=float, nargs=2, default=[0.0, 60.0])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-30.0, 30.0])
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    split_file = data_root / "lidar" / "ImageSets" / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    out_dir = resolve_path(args.output_dir)
    done = 0
    for sample_id in ids:
        save_path = out_dir / f"{sample_id}.png"
        draw_one_sample(
            data_root=data_root,
            sample_id=sample_id,
            save_path=save_path,
            x_range=(args.x_range[0], args.x_range[1]),
            y_range=(args.y_range[0], args.y_range[1]),
        )
        done += 1
        if done % 20 == 0:
            print(f"generated {done}/{len(ids)}")
    print(f"done: {done}, output: {out_dir}")


if __name__ == "__main__":
    main()

