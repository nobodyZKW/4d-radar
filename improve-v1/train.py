import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import VodRadarDatasetV1, collate_fn
from model import RadarCenterPointV1
from utils import ModelEMA, set_seed


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


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(1)
    diff = F.smooth_l1_loss(pred * m, target * m, reduction="sum")
    denom = (m.sum() * pred.shape[1]).clamp(min=1.0)
    return diff / denom


@torch.no_grad()
def _nms_center_distance(
    preds: List[Tuple[int, float, np.ndarray, np.ndarray]],
    thresh: float,
    class_agnostic: bool = False,
) -> List[Tuple[int, float, np.ndarray, np.ndarray]]:
    if len(preds) <= 1:
        return preds
    if thresh <= 0:
        return preds

    if class_agnostic:
        groups = {-1: preds}
    else:
        groups: Dict[int, List[Tuple[int, float, np.ndarray, np.ndarray]]] = {}
        for p in preds:
            groups.setdefault(p[0], []).append(p)

    kept: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
    for _, arr in groups.items():
        arr = sorted(arr, key=lambda x: x[1], reverse=True)
        suppressed = [False] * len(arr)
        for i in range(len(arr)):
            if suppressed[i]:
                continue
            pi = arr[i]
            kept.append(pi)
            for j in range(i + 1, len(arr)):
                if suppressed[j]:
                    continue
                pj = arr[j]
                d = float(np.linalg.norm(pi[2][:2] - pj[2][:2]))
                if d <= thresh:
                    suppressed[j] = True
    kept.sort(key=lambda x: x[1], reverse=True)
    return kept


@torch.no_grad()
def decode_predictions(outputs: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> List[List[Tuple[int, float, np.ndarray, np.ndarray]]]:
    hm = torch.sigmoid(outputs["heatmap"])
    if bool(cfg["decode"].get("use_local_max", True)):
        pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        hm = hm * (pooled.eq(hm).float())

    offset = outputs["offset"]
    z = outputs["z"]
    size = outputs["size"]
    yaw = outputs["yaw"]
    vel = outputs["vel"]

    b, c, h, w = hm.shape
    score_thresh = float(cfg["decode"]["score_thresh"])
    topk = int(cfg["decode"]["topk"])
    topk_mode = str(cfg["decode"].get("topk_mode", "per_class")).lower()
    nms_type = str(cfg["decode"].get("nms_type", "circle")).lower()
    nms_thresh = float(cfg["decode"].get("nms_thresh", 1.5))
    nms_class_agnostic = bool(cfg["decode"].get("nms_class_agnostic", False))

    x_min, y_min, _, _, _, _ = cfg["dataset"]["point_cloud_range"]
    vx, vy = cfg["dataset"]["voxel_size_xy"]

    out: List[List[Tuple[int, float, np.ndarray, np.ndarray]]] = []
    for bi in range(b):
        sample_preds: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
        hm_b = hm[bi]
        off_b = offset[bi]
        z_b = z[bi]
        size_b = size[bi]
        yaw_b = yaw[bi]
        vel_b = vel[bi]

        if topk_mode == "global":
            scores, inds = torch.topk(hm_b.reshape(-1), k=min(topk, hm_b.numel()))
            keep = scores > score_thresh
            scores = scores[keep]
            inds = inds[keep]
            cls_inds = inds // (h * w)
            rem = inds % (h * w)
            iy = rem // w
            ix = rem % w
            for cls_idx, s, y_i, x_i in zip(cls_inds.tolist(), scores.tolist(), iy.tolist(), ix.tolist()):
                dx = float(off_b[0, y_i, x_i])
                dy = float(off_b[1, y_i, x_i])
                zz = float(z_b[0, y_i, x_i])
                l = float(torch.exp(size_b[0, y_i, x_i]).clamp(max=30.0))
                ww = float(torch.exp(size_b[1, y_i, x_i]).clamp(max=12.0))
                hh = float(torch.exp(size_b[2, y_i, x_i]).clamp(max=10.0))
                sy = float(yaw_b[0, y_i, x_i])
                cy = float(yaw_b[1, y_i, x_i])
                yaw_rad = float(np.arctan2(sy, cy))
                vx_p = float(vel_b[0, y_i, x_i])
                vy_p = float(vel_b[1, y_i, x_i])
                x = x_min + (x_i + dx) * vx
                y = y_min + (y_i + dy) * vy
                box = np.array([x, y, zz, l, ww, hh, yaw_rad], dtype=np.float32)
                vel_xy = np.array([vx_p, vy_p], dtype=np.float32)
                sample_preds.append((int(cls_idx), float(s), box, vel_xy))
        else:
            for cls_idx in range(c):
                heat = hm_b[cls_idx]
                scores, inds = torch.topk(heat.reshape(-1), k=min(topk, heat.numel()))
                keep = scores > score_thresh
                scores = scores[keep]
                inds = inds[keep]
                for s, ind in zip(scores.tolist(), inds.tolist()):
                    y_i = int(ind // w)
                    x_i = int(ind % w)
                    dx = float(off_b[0, y_i, x_i])
                    dy = float(off_b[1, y_i, x_i])
                    zz = float(z_b[0, y_i, x_i])
                    l = float(torch.exp(size_b[0, y_i, x_i]).clamp(max=30.0))
                    ww = float(torch.exp(size_b[1, y_i, x_i]).clamp(max=12.0))
                    hh = float(torch.exp(size_b[2, y_i, x_i]).clamp(max=10.0))
                    sy = float(yaw_b[0, y_i, x_i])
                    cy = float(yaw_b[1, y_i, x_i])
                    yaw_rad = float(np.arctan2(sy, cy))
                    vx_p = float(vel_b[0, y_i, x_i])
                    vy_p = float(vel_b[1, y_i, x_i])
                    x = x_min + (x_i + dx) * vx
                    y = y_min + (y_i + dy) * vy
                    box = np.array([x, y, zz, l, ww, hh, yaw_rad], dtype=np.float32)
                    vel_xy = np.array([vx_p, vy_p], dtype=np.float32)
                    sample_preds.append((int(cls_idx), float(s), box, vel_xy))

        if nms_type in {"circle", "center"}:
            sample_preds = _nms_center_distance(sample_preds, thresh=nms_thresh, class_agnostic=nms_class_agnostic)
        out.append(sample_preds)
    return out


@torch.no_grad()
def evaluate_center_ap(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    match_dist = float(cfg["eval"]["match_center_dist"])
    num_classes = len(cfg["dataset"]["class_names"])
    stats = [{"tp": 0, "fp": 0, "gt": 0} for _ in range(num_classes)]

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True).float()
        outputs = model(bev)
        decoded = decode_predictions(outputs, cfg)
        for sample_idx, preds in enumerate(decoded):
            gt_boxes = batch["gt_boxes"][sample_idx]
            used = [False] * len(gt_boxes)
            for cls_idx, _, pred_box, _ in sorted(preds, key=lambda x: x[1], reverse=True):
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


def _forward_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    dev = outputs["heatmap"].device
    hm = batch["heatmap"].to(dev, non_blocking=True).float()
    offset = batch["offset"].to(dev, non_blocking=True).float()
    z = batch["z"].to(dev, non_blocking=True).float()
    size = batch["size"].to(dev, non_blocking=True).float()
    yaw = batch["yaw"].to(dev, non_blocking=True).float()
    vel = batch["vel"].to(dev, non_blocking=True).float()
    reg_mask = batch["reg_mask"].to(dev, non_blocking=True).float()
    vel_mask = batch["vel_mask"].to(dev, non_blocking=True).float()

    loss_hm = focal_heatmap_loss(outputs["heatmap"], hm)
    loss_offset = masked_smooth_l1(outputs["offset"], offset, reg_mask)
    loss_z = masked_smooth_l1(outputs["z"], z, reg_mask)
    loss_size = masked_smooth_l1(outputs["size"], size, reg_mask)
    loss_box = loss_offset + loss_z + loss_size
    loss_yaw = masked_smooth_l1(outputs["yaw"], yaw, reg_mask)

    use_vel_loss = bool(cfg["train"].get("use_velocity_loss", False)) and bool(cfg["model"].get("use_velocity_head", True))
    if use_vel_loss:
        loss_vel = masked_smooth_l1(outputs["vel"], vel, vel_mask)
    else:
        loss_vel = torch.zeros((), device=outputs["heatmap"].device)

    hm_w = float(cfg["train"].get("hm_weight", 1.0))
    box_w = float(cfg["train"].get("box_weight", 2.0))
    yaw_w = float(cfg["train"].get("yaw_weight", 0.2))
    vel_w = float(cfg["train"].get("vel_weight", 0.2))

    total = hm_w * loss_hm + box_w * loss_box + yaw_w * loss_yaw + vel_w * loss_vel
    return {
        "loss_total": total,
        "loss_hm": loss_hm,
        "loss_box": loss_box,
        "loss_yaw": loss_yaw,
        "loss_vel": loss_vel,
        "loss_offset": loss_offset,
        "loss_z": loss_z,
        "loss_size": loss_size,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    ema: ModelEMA | None = None,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.train()
    meters = {
        "loss_total": 0.0,
        "loss_hm": 0.0,
        "loss_box": 0.0,
        "loss_yaw": 0.0,
        "loss_vel": 0.0,
    }
    steps = 0
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 0.0))

    pbar = tqdm(loader, desc="train", leave=False)
    for bi, batch in enumerate(pbar):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True).float()
        outputs = model(bev)
        loss_dict = _forward_losses(outputs, batch, cfg)
        loss = loss_dict["loss_total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        for k in meters:
            meters[k] += float(loss_dict[k].item())
        steps += 1
        pbar.set_postfix(
            loss=f"{loss_dict['loss_total'].item():.4f}",
            hm=f"{loss_dict['loss_hm'].item():.4f}",
            box=f"{loss_dict['loss_box'].item():.4f}",
            yaw=f"{loss_dict['loss_yaw'].item():.4f}",
            vel=f"{loss_dict['loss_vel'].item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    out = {k: v / max(steps, 1) for k, v in meters.items()}
    out["lr"] = float(optimizer.param_groups[0]["lr"])
    return out


@torch.no_grad()
def val_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    meters = {
        "loss_total": 0.0,
        "loss_hm": 0.0,
        "loss_box": 0.0,
        "loss_yaw": 0.0,
        "loss_vel": 0.0,
    }
    steps = 0
    for bi, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        bev = batch["bev"].to(device, non_blocking=True).float()
        outputs = model(bev)
        loss_dict = _forward_losses(outputs, batch, cfg)
        for k in meters:
            meters[k] += float(loss_dict[k].item())
        steps += 1
    return {k: v / max(steps, 1) for k, v in meters.items()}


def build_model(cfg: Dict[str, Any]) -> RadarCenterPointV1:
    in_channels = int(cfg["dataset"].get("num_input_channels", cfg["model"]["in_channels"]))
    return RadarCenterPointV1(
        in_channels=in_channels,
        num_classes=len(cfg["dataset"]["class_names"]),
        use_motion_branch=bool(cfg["model"].get("use_motion_branch", True)),
        geometry_channels=list(cfg["model"].get("geometry_channels", [])),
        motion_channels=list(cfg["model"].get("motion_channels", [])),
        base_channels=int(cfg["model"].get("base_channels", 32)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], epochs: int):
    sch_cfg = cfg["train"].get("scheduler", {})
    sch_type = str(sch_cfg.get("type", "none")).lower()
    if sch_type == "none":
        return None
    if sch_type == "cosine":
        min_lr = float(sch_cfg.get("min_lr", 1e-5))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if sch_type == "multistep":
        milestones = [int(x) for x in sch_cfg.get("milestones", [15, 25])]
        gamma = float(sch_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    raise ValueError(f"Unsupported scheduler type: {sch_type}")


def main() -> None:
    parser = argparse.ArgumentParser("4D Radar CenterPoint-Radar v1 Training")
    parser.add_argument("--config", type=str, default="improve-v1/configs/vod_centerpoint_radar_v1.yaml")
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

    train_set = VodRadarDatasetV1(cfg, split=cfg["dataset"]["train_split"], training=True)
    val_set = VodRadarDatasetV1(cfg, split=cfg["dataset"]["val_split"], training=False)
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

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"] if args.epochs <= 0 else args.epochs)
    scheduler = build_scheduler(optimizer, cfg, epochs=epochs)

    ema: ModelEMA | None = None
    ema_cfg = cfg["train"].get("ema", {})
    if bool(ema_cfg.get("use_ema", False)):
        ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))

    best_f1 = -1.0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            ema=ema,
            max_batches=args.max_train_batches,
        )

        eval_model = model
        if ema is not None and bool(ema_cfg.get("eval_with_ema", True)):
            eval_model = ema.ema.to(device)

        val_stats = val_losses(eval_model, val_loader, device, cfg, max_batches=args.max_val_batches)
        eval_stats = evaluate_center_ap(eval_model, val_loader, device, cfg, max_batches=args.max_val_batches)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            **train_stats,
            **{f"val_{k}": v for k, v in val_stats.items()},
            **eval_stats,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_stats['loss_total']:.4f} "
            f"hm={train_stats['loss_hm']:.4f} "
            f"box={train_stats['loss_box']:.4f} "
            f"yaw={train_stats['loss_yaw']:.4f} "
            f"vel={train_stats['loss_vel']:.4f} "
            f"val_loss={val_stats['loss_total']:.4f} "
            f"mean_f1={eval_stats['mean_f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        print(json.dumps(eval_stats, ensure_ascii=False))

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
        }
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        if ema is not None:
            ckpt["ema_model"] = ema.state_dict()

        last_ckpt = output_dir / "last.pt"
        torch.save(ckpt, last_ckpt)
        if eval_stats["mean_f1"] > best_f1:
            best_f1 = eval_stats["mean_f1"]
            best_ckpt = output_dir / "best.pt"
            torch.save(ckpt, best_ckpt)

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training complete. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
