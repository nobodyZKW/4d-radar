import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from baseline.dataset import VodRadarDataset, collate_fn
    from baseline.model import RadarBaselineNet
    from baseline.utils import set_seed
else:
    from .dataset import VodRadarDataset, collate_fn
    from .model import RadarBaselineNet
    from .utils import set_seed


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = resolve_path(str(config_path))
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["dataset"]["root"] = str((config_path.parent / cfg["dataset"]["root"]).resolve())
    return cfg


def focal_heatmap_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits).clamp(min=1e-4, max=1 - 1e-4)
    pos_inds = target.eq(1.0).float()
    neg_inds = target.lt(1.0).float()
    neg_weights = torch.pow(1.0 - target, 4.0)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos <= 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def reg_l1_loss(pred_reg: torch.Tensor, target_reg: torch.Tensor, reg_mask: torch.Tensor) -> torch.Tensor:
    mask = reg_mask.unsqueeze(1)
    diff = F.smooth_l1_loss(pred_reg * mask, target_reg * mask, reduction="sum")
    denom = mask.sum().clamp(min=1.0)
    return diff / denom


@torch.no_grad()
def decode_predictions(
    pred_hm: torch.Tensor,
    pred_reg: torch.Tensor,
    score_thresh: float,
    topk: int,
    x_min: float,
    y_min: float,
    vx: float,
    vy: float,
) -> List[List[Tuple[int, float, np.ndarray]]]:
    b, c, h, w = pred_hm.shape
    hm = torch.sigmoid(pred_hm)
    outputs: List[List[Tuple[int, float, np.ndarray]]] = []
    for bi in range(b):
        per_sample: List[Tuple[int, float, np.ndarray]] = []
        hm_b = hm[bi]
        reg_b = pred_reg[bi]
        for cls_idx in range(c):
            heat = hm_b[cls_idx]
            scores, inds = torch.topk(heat.reshape(-1), k=min(topk, heat.numel()))
            keep = scores > score_thresh
            scores = scores[keep]
            inds = inds[keep]
            for s, ind in zip(scores.tolist(), inds.tolist()):
                iy = int(ind // w)
                ix = int(ind % w)
                dx = float(reg_b[0, iy, ix])
                dy = float(reg_b[1, iy, ix])
                z = float(reg_b[2, iy, ix])
                l = float(torch.exp(reg_b[3, iy, ix]).clamp(max=20.0))
                ww = float(torch.exp(reg_b[4, iy, ix]).clamp(max=10.0))
                hh = float(torch.exp(reg_b[5, iy, ix]).clamp(max=8.0))
                sin_yaw = float(reg_b[6, iy, ix])
                cos_yaw = float(reg_b[7, iy, ix])
                yaw = float(np.arctan2(sin_yaw, cos_yaw))
                x = x_min + (ix + dx) * vx
                y = y_min + (iy + dy) * vy
                box = np.array([x, y, z, l, ww, hh, yaw], dtype=np.float32)
                per_sample.append((cls_idx, float(s), box))
        outputs.append(per_sample)
    return outputs


@torch.no_grad()
def evaluate_center_ap(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    x_min, y_min, _, _, _, _ = cfg["dataset"]["point_cloud_range"]
    vx, vy = cfg["dataset"]["voxel_size_xy"]
    score_thresh = float(cfg["eval"]["score_thresh"])
    topk = int(cfg["eval"]["topk_per_class"])
    match_dist = float(cfg["eval"]["match_center_dist"])
    num_classes = len(cfg["dataset"]["class_names"])

    stats = [{"tp": 0, "fp": 0, "gt": 0} for _ in range(num_classes)]
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        outputs = model(bev)
        decoded = decode_predictions(outputs["heatmap"], outputs["reg"], score_thresh, topk, x_min, y_min, vx, vy)

        for sample_idx, preds in enumerate(decoded):
            gt_boxes = batch["gt_boxes"][sample_idx]
            used = [False] * len(gt_boxes)
            for cls_idx, _, pred_box in sorted(preds, key=lambda x: x[1], reverse=True):
                best_j = -1
                best_d = 1e9
                for j, (gt_cls, gt_box) in enumerate(gt_boxes):
                    if used[j] or gt_cls != cls_idx:
                        continue
                    d = float(np.linalg.norm(pred_box[:2] - gt_box[:2]))
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d <= match_dist:
                    used[best_j] = True
                    stats[cls_idx]["tp"] += 1
                else:
                    stats[cls_idx]["fp"] += 1
            for gt_cls, _ in gt_boxes:
                stats[gt_cls]["gt"] += 1

    out: Dict[str, float] = {}
    mean_f1 = 0.0
    for cls_idx, name in enumerate(cfg["dataset"]["class_names"]):
        tp = stats[cls_idx]["tp"]
        fp = stats[cls_idx]["fp"]
        gt = stats[cls_idx]["gt"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        mean_f1 += f1
    out["mean_f1"] = mean_f1 / max(num_classes, 1)
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    hm_w: float,
    reg_w: float,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    hm_meter = 0.0
    reg_meter = 0.0
    total_meter = 0.0
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for bi, batch in enumerate(pbar):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        tgt_hm = batch["heatmap"].to(device, non_blocking=True)
        tgt_reg = batch["reg"].to(device, non_blocking=True)
        reg_mask = batch["reg_mask"].to(device, non_blocking=True)

        outputs = model(bev)
        loss_hm = focal_heatmap_loss(outputs["heatmap"], tgt_hm)
        loss_reg = reg_l1_loss(outputs["reg"], tgt_reg, reg_mask)
        loss = hm_w * loss_hm + reg_w * loss_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        hm_meter += float(loss_hm.item())
        reg_meter += float(loss_reg.item())
        total_meter += float(loss.item())
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", hm=f"{loss_hm.item():.4f}", reg=f"{loss_reg.item():.4f}")

    return {
        "loss": total_meter / max(steps, 1),
        "loss_hm": hm_meter / max(steps, 1),
        "loss_reg": reg_meter / max(steps, 1),
    }


@torch.no_grad()
def val_losses(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    hm_meter = 0.0
    reg_meter = 0.0
    total_meter = 0.0
    steps = 0
    for bi, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True)
        tgt_hm = batch["heatmap"].to(device, non_blocking=True)
        tgt_reg = batch["reg"].to(device, non_blocking=True)
        reg_mask = batch["reg_mask"].to(device, non_blocking=True)
        outputs = model(bev)
        loss_hm = focal_heatmap_loss(outputs["heatmap"], tgt_hm)
        loss_reg = reg_l1_loss(outputs["reg"], tgt_reg, reg_mask)
        loss = loss_hm + loss_reg
        hm_meter += float(loss_hm.item())
        reg_meter += float(loss_reg.item())
        total_meter += float(loss.item())
        steps += 1
    return {
        "loss": total_meter / max(steps, 1),
        "loss_hm": hm_meter / max(steps, 1),
        "loss_reg": reg_meter / max(steps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar Baseline Training")
    parser.add_argument("--config", type=str, default="baseline/configs/vod_baseline.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
    output_dir = (config_path.parent / cfg["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = VodRadarDataset(cfg, split=cfg["dataset"]["train_split"])
    val_set = VodRadarDataset(cfg, split=cfg["dataset"]["val_split"])
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"] if args.epochs <= 0 else args.epochs)
    hm_w = float(cfg["train"]["hm_loss_weight"])
    reg_w = float(cfg["train"]["reg_loss_weight"])

    best_f1 = -1.0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, hm_w, reg_w, max_batches=args.max_train_batches
        )
        val_stats = val_losses(model, val_loader, device, max_batches=args.max_val_batches)
        eval_stats = evaluate_center_ap(model, val_loader, device, cfg, max_batches=args.max_val_batches)

        row = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_stats.items()}, **eval_stats}
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"mean_f1={eval_stats['mean_f1']:.4f}"
        )
        print(json.dumps(eval_stats, ensure_ascii=False))

        last_ckpt = output_dir / "last.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg}, last_ckpt)
        if eval_stats["mean_f1"] > best_f1:
            best_f1 = eval_stats["mean_f1"]
            best_ckpt = output_dir / "best.pt"
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg}, best_ckpt
            )

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
