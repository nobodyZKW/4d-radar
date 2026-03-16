import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import VodRadarDatasetV1, collate_fn
from train import build_model, evaluate_center_ap, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar CenterPoint-Radar v1 Evaluation")
    parser.add_argument("--config", type=str, default="improve-v1/configs/vod_centerpoint_radar_v1.yaml")
    parser.add_argument("--ckpt", type=str, default="improve-v1/outputs/vod_centerpoint_radar_v1/best.pt")
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    ckpt_path = resolve_path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_set = VodRadarDatasetV1(cfg, split=cfg["dataset"]["val_split"], training=False)
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_ema = bool(args.use_ema)
    if use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)

    stats = evaluate_center_ap(model, val_loader, device, cfg, max_batches=args.max_val_batches)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

