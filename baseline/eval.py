import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from baseline.dataset import VodRadarDataset, collate_fn
    from baseline.model import RadarBaselineNet
    from baseline.train import evaluate_center_ap, load_config, resolve_path
else:
    from .dataset import VodRadarDataset, collate_fn
    from .model import RadarBaselineNet
    from .train import evaluate_center_ap, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar Baseline Evaluation")
    parser.add_argument("--config", type=str, default="baseline/configs/vod_baseline.yaml")
    parser.add_argument("--ckpt", type=str, default="baseline/outputs/vod_baseline/best.pt")
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    ckpt_path = resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_set = VodRadarDataset(cfg, split=cfg["dataset"]["val_split"])
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = RadarBaselineNet(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=len(cfg["dataset"]["class_names"]),
    ).to(device)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    stats = evaluate_center_ap(model, val_loader, device, cfg, max_batches=args.max_val_batches)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
