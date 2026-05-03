from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_EXP_IDS = ["vel_none", "vel_weak_heading", "vel_robust_lstsq"]
VEL_PAIR_PATTERN = re.compile(r"([A-Za-z0-9_/]+)=([-+0-9.eE]+)")


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_experiment_rows(input_dir: Path, exp_ids: List[str]) -> List[Dict]:
    summary_csv = input_dir / "ablation_summary.csv"
    csv_rows_by_id: Dict[str, Dict] = {}
    if summary_csv.exists():
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_id = str(row.get("id", "")).strip()
                if exp_id:
                    csv_rows_by_id[exp_id] = row

    rows: List[Dict] = []
    for exp_id in exp_ids:
        json_path = input_dir / f"{exp_id}.json"
        if not json_path.exists():
            if exp_id in csv_rows_by_id:
                row = dict(csv_rows_by_id[exp_id])
                row["missing"] = False
                rows.append(row)
            else:
                rows.append({"id": exp_id, "missing": True})
            continue
        row = json.loads(json_path.read_text(encoding="utf-8"))
        row["missing"] = False
        rows.append(row)
    return rows


def _resolve_train_log(root: Path, row: Dict) -> Optional[Path]:
    cfg_rel = str(row.get("cfg", ""))
    extra_tag = str(row.get("extra_tag", ""))
    if not cfg_rel or not extra_tag:
        return None

    cfg_path = (root / cfg_rel).resolve()
    exp_group = cfg_path.parent.name
    cfg_name = cfg_path.stem
    out_dir = root / "external" / "OpenPCDet" / "output" / exp_group / cfg_name / extra_tag
    logs = sorted(out_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _extract_vel_stats_from_log(log_path: Optional[Path]) -> Dict[str, float]:
    if log_path is None or not log_path.exists():
        return {}

    latest_stats: Dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "VelStats:" not in line:
            continue
        tail = line.split("VelStats:", 1)[1]
        pairs = VEL_PAIR_PATTERN.findall(tail)
        if not pairs:
            continue
        latest_stats = {k: _to_float(v) for k, v in pairs}
    return latest_stats


def _pick_official_key(keys: List[str], class_name: str) -> Optional[str]:
    cls_keys = [k for k in keys if k.startswith(class_name)]
    if not cls_keys:
        return None

    preferred = [k for k in cls_keys if "3d" in k.lower() and "r40" in k.lower() and "moderate" in k.lower()]
    if preferred:
        return sorted(preferred)[0]

    preferred = [k for k in cls_keys if "3d" in k.lower() and "r40" in k.lower()]
    if preferred:
        return sorted(preferred)[0]

    preferred = [k for k in cls_keys if "3d" in k.lower()]
    if preferred:
        return sorted(preferred)[0]

    return sorted(cls_keys)[0]


def _build_summary_rows(root: Path, rows: List[Dict]) -> List[Dict]:
    summary_rows: List[Dict] = []
    for row in rows:
        if row.get("missing", False):
            summary_rows.append(
                {
                    "id": row.get("id", "unknown"),
                    "status": "missing_result",
                }
            )
            continue

        train_log = _resolve_train_log(root, row)
        vel_log_stats = _extract_vel_stats_from_log(train_log)

        numeric_keys = [k for k, v in row.items() if isinstance(v, (int, float))]
        car_official_key = _pick_official_key(numeric_keys, "Car")
        ped_official_key = _pick_official_key(numeric_keys, "Pedestrian")
        cyc_official_key = _pick_official_key(numeric_keys, "Cyclist")

        car_official = _to_float(row.get(car_official_key, 0.0)) if car_official_key else 0.0
        ped_official = _to_float(row.get(ped_official_key, 0.0)) if ped_official_key else 0.0
        cyc_official = _to_float(row.get(cyc_official_key, 0.0)) if cyc_official_key else 0.0
        official_mean = (car_official + ped_official + cyc_official) / 3.0

        summary = {
            "id": str(row.get("id", "")),
            "status": "dry-run" if str(row.get("note", "")).strip().lower() == "dry-run" else "ok",
            "cfg": str(row.get("cfg", "")),
            "extra_tag": str(row.get("extra_tag", "")),
            "ckpt": str(row.get("ckpt", "")),
            "train_log": str(train_log) if train_log is not None else "",
            "quick/mean_f1": _to_float(row.get("quick/mean_f1", 0.0)),
            "quick/Car_f1": _to_float(row.get("quick/Car_f1", 0.0)),
            "quick/Pedestrian_f1": _to_float(row.get("quick/Pedestrian_f1", 0.0)),
            "quick/Cyclist_f1": _to_float(row.get("quick/Cyclist_f1", 0.0)),
            "official/Car_3d": car_official,
            "official/Pedestrian_3d": ped_official,
            "official/Cyclist_3d": cyc_official,
            "official/mean_3d": official_mean,
            "official/Car_key": car_official_key or "",
            "official/Pedestrian_key": ped_official_key or "",
            "official/Cyclist_key": cyc_official_key or "",
            "vel_loss": _to_float(vel_log_stats.get("vel_loss", 0.0)),
            "num_valid_vel_boxes": _to_float(vel_log_stats.get("num_valid_vel_boxes", 0.0)),
            "num_weak_vel_boxes": _to_float(vel_log_stats.get("num_weak_vel_boxes", 0.0)),
            "velocity_branch_activation_ratio": _to_float(vel_log_stats.get("velocity_branch_activation_ratio", 0.0)),
            "vel_fit_residual_mean": _to_float(vel_log_stats.get("vel_fit_residual_mean", 0.0)),
        }
        summary_rows.append(summary)
    return summary_rows


def _write_csv(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: List[Dict], output_md: Path) -> None:
    lines = [
        "# Velocity Supervision Validation Summary",
        "",
        "| id | status | quick/mean_f1 | official/mean_3d | vel_loss | activation_ratio | valid_vel_boxes | weak_vel_boxes | residual_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {id} | {status} | {q:.4f} | {o:.4f} | {vl:.4f} | {ar:.4f} | {nv:.1f} | {nw:.1f} | {rm:.4f} |".format(
                id=r.get("id", ""),
                status=r.get("status", ""),
                q=_to_float(r.get("quick/mean_f1", 0.0)),
                o=_to_float(r.get("official/mean_3d", 0.0)),
                vl=_to_float(r.get("vel_loss", 0.0)),
                ar=_to_float(r.get("velocity_branch_activation_ratio", 0.0)),
                nv=_to_float(r.get("num_valid_vel_boxes", 0.0)),
                nw=_to_float(r.get("num_weak_vel_boxes", 0.0)),
                rm=_to_float(r.get("vel_fit_residual_mean", 0.0)),
            )
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(rows: List[Dict], output_png: Path) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        note_path = output_png.with_suffix(".txt")
        note_path.write_text(
            "matplotlib not available, skip plot generation.\n"
            f"reason: {e}\n"
            "Install matplotlib to generate velocity validation chart.",
            encoding="utf-8",
        )
        return f"plot skipped -> {note_path}"

    valid_rows = [r for r in rows if r.get("status") == "ok"]
    if not valid_rows:
        note_path = output_png.with_suffix(".txt")
        note_path.write_text("No valid rows found, skip plot generation.", encoding="utf-8")
        return f"plot skipped -> {note_path}"

    labels = [str(r.get("id", "")) for r in valid_rows]
    quick_mean_f1 = np.array([_to_float(r.get("quick/mean_f1", 0.0)) for r in valid_rows], dtype=np.float32)
    official_mean_3d = np.array([_to_float(r.get("official/mean_3d", 0.0)) for r in valid_rows], dtype=np.float32)
    activation = np.array(
        [_to_float(r.get("velocity_branch_activation_ratio", 0.0)) for r in valid_rows], dtype=np.float32
    )

    x = np.arange(len(labels))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x - width / 2, quick_mean_f1, width=width, label="quick/mean_f1")
    axes[0].bar(x + width / 2, official_mean_3d, width=width, label="official/mean_3d")
    axes[0].set_title("Velocity Supervision: Quick vs Official")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, activation, width=0.45, color="#2b8a3e")
    axes[1].set_title("Velocity Branch Activation Ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return f"plot saved -> {output_png}"


def main():
    parser = argparse.ArgumentParser("Build velocity supervision validation summary")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--input-dir", type=str, default="outputs/ablations/velocity")
    parser.add_argument("--exp-ids", type=str, default=",".join(DEFAULT_EXP_IDS))
    parser.add_argument("--output-csv", type=str, default="outputs/ablations/velocity/velocity_validation_summary.csv")
    parser.add_argument("--output-md", type=str, default="outputs/ablations/velocity/velocity_validation_summary.md")
    parser.add_argument(
        "--output-plot", type=str, default="outputs/ablations/velocity/velocity_validation_summary.png"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_dir = (root / args.input_dir).resolve()
    exp_ids = [x.strip() for x in args.exp_ids.split(",") if x.strip()]

    rows = _load_experiment_rows(input_dir=input_dir, exp_ids=exp_ids)
    summary_rows = _build_summary_rows(root=root, rows=rows)

    output_csv = (root / args.output_csv).resolve()
    output_md = (root / args.output_md).resolve()
    output_plot = (root / args.output_plot).resolve()
    _write_csv(summary_rows, output_csv)
    _write_md(summary_rows, output_md)
    plot_msg = _write_plot(summary_rows, output_plot)

    print(f"Saved csv: {output_csv}")
    print(f"Saved md: {output_md}")
    print(plot_msg)


if __name__ == "__main__":
    main()
