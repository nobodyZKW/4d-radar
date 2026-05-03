from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


DEFAULT_EXPERIMENTS = [
    {"id": "A0", "cfg": "configs/model_configs/ablation_a0_dynpillar_baseline.yaml", "set": []},
    {"id": "A1", "cfg": "configs/model_configs/ablation_a1_radar7pillar.yaml", "set": []},
    {"id": "A2", "cfg": "configs/model_configs/ablation_a2_radar7pillar_vel.yaml", "set": []},
    {"id": "A3", "cfg": "configs/model_configs/ablation_a3_radar7pillar_vel_decode_tuned.yaml", "set": []},
    {"id": "A4", "cfg": "configs/model_configs/ablation_a4_full.yaml", "set": []},
    {
        "id": "feat_xyz_only",
        "cfg": "configs/model_configs/ablation_a4_full.yaml",
        "set": [
            "MODEL.VFE.USE_RCS", "False",
            "MODEL.VFE.USE_VR", "False",
            "MODEL.VFE.USE_VR_COMP", "False",
            "MODEL.VFE.USE_TIME", "False",
        ],
    },
    {
        "id": "feat_xyz_rcs",
        "cfg": "configs/model_configs/ablation_a4_full.yaml",
        "set": [
            "MODEL.VFE.USE_RCS", "True",
            "MODEL.VFE.USE_VR", "False",
            "MODEL.VFE.USE_VR_COMP", "False",
            "MODEL.VFE.USE_TIME", "False",
        ],
    },
    {
        "id": "feat_xyz_rcs_vr",
        "cfg": "configs/model_configs/ablation_a4_full.yaml",
        "set": [
            "MODEL.VFE.USE_RCS", "True",
            "MODEL.VFE.USE_VR", "True",
            "MODEL.VFE.USE_VR_COMP", "False",
            "MODEL.VFE.USE_TIME", "False",
        ],
    },
    {
        "id": "feat_xyz_rcs_vr_vrcomp",
        "cfg": "configs/model_configs/ablation_a4_full.yaml",
        "set": [
            "MODEL.VFE.USE_RCS", "True",
            "MODEL.VFE.USE_VR", "True",
            "MODEL.VFE.USE_VR_COMP", "True",
            "MODEL.VFE.USE_TIME", "False",
        ],
    },
    {
        "id": "vel_none",
        "cfg": "configs/model_configs/ablation_a1_radar7pillar.yaml",
        "set": [],
    },
    {
        "id": "vel_weak_heading",
        "cfg": "configs/model_configs/ablation_a2_radar7pillar_vel.yaml",
        "set": [
            "DATA_CONFIG.VELOCITY_SUPERVISION.SUPERVISION_MODE", "weak_only",
            "DATA_CONFIG.VELOCITY_SUPERVISION.USE_FALLBACK_HEADING", "True",
            "DATA_CONFIG.VELOCITY_SUPERVISION.WEAK_WEIGHT", "0.25",
        ],
    },
    {
        "id": "vel_robust_lstsq",
        "cfg": "configs/model_configs/ablation_a2_radar7pillar_vel.yaml",
        "set": [
            "DATA_CONFIG.VELOCITY_SUPERVISION.SUPERVISION_MODE", "robust",
            "DATA_CONFIG.VELOCITY_SUPERVISION.MAX_CONDITION", "5000.0",
            "DATA_CONFIG.VELOCITY_SUPERVISION.USE_FALLBACK_HEADING", "True",
        ],
    },
]


PRESET_EXP_IDS = {
    "a0_a2": ["A0", "A1", "A2"],
    "a0_a4": ["A0", "A1", "A2", "A3", "A4"],
    "feature": ["feat_xyz_only", "feat_xyz_rcs", "feat_xyz_rcs_vr", "feat_xyz_rcs_vr_vrcomp"],
    "velocity": ["vel_none", "vel_weak_heading", "vel_robust_lstsq"],
}


def resolve_paths(root: Path, cfg_rel: str, extra_tag: str, set_cfgs: List[str]) -> Tuple[Path, List[Path]]:
    cfg_path = (root / cfg_rel).resolve()
    cfg_obj = copy.deepcopy(cfg)
    cfg_from_yaml_file(str(cfg_path), cfg_obj)
    if set_cfgs:
        cfg_from_list(set_cfgs, cfg_obj)

    output_root = root / "external" / "OpenPCDet" / "output"
    cfg_tag = str(cfg_obj.get("TAG", cfg_path.stem))
    exp_group = str(cfg_obj.get("EXP_GROUP_PATH", "")).strip().strip("/\\")

    candidates = []
    if exp_group:
        candidates.append(output_root / exp_group / cfg_tag / extra_tag / "ckpt")
    # OpenPCDet in this repo often writes to output/<TAG>/<extra_tag>/ckpt
    candidates.append(output_root / cfg_tag / extra_tag / "ckpt")
    # Backward-compatible fallback for old runner logic.
    candidates.append(output_root / cfg_path.parent.name / cfg_path.stem / extra_tag / "ckpt")

    deduped = []
    for p in candidates:
        if p not in deduped:
            deduped.append(p)
    return cfg_path, deduped


def find_latest_ckpt(ckpt_dirs: List[Path]) -> Path:
    checked = []
    for ckpt_dir in ckpt_dirs:
        checked.append(str(ckpt_dir))
        if not ckpt_dir.exists():
            continue
        ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return ckpts[-1]
    checked_str = "\n".join(checked)
    raise FileNotFoundError(f"No checkpoint found in candidate dirs:\n{checked_str}")
    return ckpts[-1]


def run_train(root: Path, cfg_path: Path, extra_tag: str, workers: int, epochs: int, set_cfgs: List[str]) -> None:
    cmd = [
        sys.executable,
        str(root / "external" / "OpenPCDet" / "tools" / "train.py"),
        "--cfg_file",
        str(cfg_path),
        "--extra_tag",
        extra_tag,
        "--workers",
        str(workers),
    ]
    if epochs > 0:
        cmd.extend(["--epochs", str(epochs)])
    if set_cfgs:
        cmd.append("--set")
        cmd.extend(set_cfgs)
    subprocess.run(cmd, cwd=str(root), check=True)


@torch.no_grad()
def evaluate_cfg_ckpt(root: Path, cfg_path: Path, ckpt_path: Path, workers: int, set_cfgs: List[str]) -> Dict:
    cfg_obj = copy.deepcopy(cfg)
    cfg_from_yaml_file(str(cfg_path), cfg_obj)
    if set_cfgs:
        cfg_from_list(set_cfgs, cfg_obj)

    logger = common_utils.create_logger()
    _, loader, _ = build_dataloader(
        dataset_cfg=cfg_obj.DATA_CONFIG,
        class_names=cfg_obj.CLASS_NAMES,
        batch_size=int(cfg_obj.OPTIMIZATION.get("BATCH_SIZE_PER_GPU", 2)),
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg_obj.MODEL, num_class=len(cfg_obj.CLASS_NAMES), dataset=loader.dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    det_annos = []
    for batch_dict in loader:
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model(batch_dict)
        annos = loader.dataset.generate_prediction_dicts(batch_dict, pred_dicts, loader.dataset.class_names, output_path=None)
        det_annos.extend(annos)

    _, metrics = loader.dataset.evaluation(
        det_annos,
        loader.dataset.class_names,
        eval_metric=cfg_obj.MODEL.POST_PROCESSING.EVAL_METRIC,
    )
    return metrics


def select_experiments(experiments: List[Dict], exp_ids: List[str]) -> List[Dict]:
    if not exp_ids:
        return experiments

    exp_map = {str(x["id"]): x for x in experiments}
    selected = []
    missing = []
    for exp_id in exp_ids:
        if exp_id in exp_map:
            selected.append(exp_map[exp_id])
        else:
            missing.append(exp_id)
    if missing:
        raise ValueError(f"Unknown experiment ids: {missing}")
    return selected


def write_summary(rows: List[Dict], csv_path: Path, md_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Ablation Summary", "", "| id | cfg | quick/mean_f1 | ckpt |", "|---|---|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r.get('id','')} | `{r.get('cfg','')}` | {float(r.get('quick/mean_f1', 0.0)):.4f} | `{r.get('ckpt','')}` |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("Run ablation suite for improve-v1.5")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--extra-tag-prefix", type=str, default="ablation")
    parser.add_argument("--output-dir", type=str, default="outputs/ablations")
    parser.add_argument("--experiments-json", type=str, default="")
    parser.add_argument("--preset", type=str, default="")
    parser.add_argument("--exp-ids", type=str, default="")
    parser.add_argument("--skip-missing-ckpt", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiments_json:
        experiments = json.loads(Path(args.experiments_json).read_text(encoding="utf-8"))
    else:
        experiments = DEFAULT_EXPERIMENTS

    if args.preset:
        preset_key = args.preset.strip().lower()
        if preset_key not in PRESET_EXP_IDS:
            raise ValueError(f"Unknown preset: {args.preset}. Available presets: {sorted(PRESET_EXP_IDS.keys())}")
        experiments = select_experiments(experiments, PRESET_EXP_IDS[preset_key])

    if args.exp_ids.strip():
        exp_ids = [x.strip() for x in args.exp_ids.split(",") if x.strip()]
        experiments = select_experiments(experiments, exp_ids)

    if args.list:
        print("Available experiment ids:")
        for exp in experiments:
            print(f"- {exp['id']}: {exp['cfg']}")
        return

    rows = []
    for exp in experiments:
        exp_id = exp["id"]
        cfg_rel = exp["cfg"]
        set_cfgs = list(exp.get("set", []))
        extra_tag = f"{args.extra_tag_prefix}_{exp_id}"

        cfg_path, ckpt_dirs = resolve_paths(root, cfg_rel, extra_tag, set_cfgs)
        print(f"[Ablation] {exp_id} -> cfg={cfg_path}")
        if args.dry_run:
            rows.append(
                {
                    "id": exp_id,
                    "cfg": cfg_rel,
                    "ckpt": "",
                    "quick/mean_f1": 0.0,
                    "note": "dry-run",
                    "extra_tag": extra_tag,
                    "expected_ckpt_dirs": " | ".join(str(x) for x in ckpt_dirs),
                }
            )
            continue

        if args.run_train:
            run_train(root=root, cfg_path=cfg_path, extra_tag=extra_tag, workers=args.workers, epochs=args.epochs, set_cfgs=set_cfgs)

        try:
            ckpt_path = find_latest_ckpt(ckpt_dirs)
        except FileNotFoundError:
            if not args.skip_missing_ckpt:
                raise
            row = {
                "id": exp_id,
                "cfg": cfg_rel,
                "ckpt": "",
                "quick/mean_f1": 0.0,
                "note": "missing-ckpt",
                "extra_tag": extra_tag,
                "expected_ckpt_dirs": " | ".join(str(x) for x in ckpt_dirs),
            }
            rows.append(row)
            json_path = output_dir / f"{exp_id}.json"
            json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[WARN] Missing checkpoint for {exp_id}, skipped. Expected one of:\n" + "\n".join(str(x) for x in ckpt_dirs))
            continue

        metrics = evaluate_cfg_ckpt(root=root, cfg_path=cfg_path, ckpt_path=ckpt_path, workers=args.workers, set_cfgs=set_cfgs)

        row = {
            "id": exp_id,
            "cfg": cfg_rel,
            "ckpt": str(ckpt_path),
            "extra_tag": extra_tag,
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        rows.append(row)

        json_path = output_dir / f"{exp_id}.json"
        json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}")

    summary_csv = output_dir / "ablation_summary.csv"
    summary_md = output_dir / "ablation_summary.md"
    write_summary(rows, summary_csv, summary_md)
    print(f"Saved summary csv: {summary_csv}")
    print(f"Saved summary md: {summary_md}")


if __name__ == "__main__":
    main()
