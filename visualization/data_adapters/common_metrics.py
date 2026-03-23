from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def to_optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return {}

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        # YAML is a superset of JSON, fallback for minimal environments.
        try:
            return json.loads(text)
        except Exception:
            return {}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists() or not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_history_json(path: Path) -> Dict[str, Any]:
    history = load_json(path, []) if path.exists() else []
    if not isinstance(history, list):
        history = []

    latest = history[-1] if history else {}
    best = {}
    if history:
        best = max(history, key=lambda x: safe_float(x.get("mean_f1", -1.0), -1.0))

    curves = {
        "epoch": [int(x.get("epoch", i + 1)) for i, x in enumerate(history)],
        "mean_f1": [safe_float(x.get("mean_f1", 0.0)) for x in history],
        "loss": [safe_float(x.get("loss_total", x.get("loss", 0.0))) for x in history],
        "val_loss": [safe_float(x.get("val_loss_total", x.get("val_loss", 0.0))) for x in history],
    }

    return {
        "available": path.exists(),
        "path": str(path),
        "history": history,
        "latest": latest,
        "best": best,
        "curves": curves,
    }


def parse_openpcdet_train_log(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "train_curve": [],
            "quick_metrics": {},
            "official_rows": [],
            "official_summary": {},
            "recall": {},
        }

    train_re = re.compile(
        r"Train:\s+(\d+)/(\d+).*?\[\s*(\d+)/(\d+).*?\]\s+Loss:\s+([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\)\s+LR:\s+([0-9.eE+-]+).*?Acc_iter\s+(\d+)"
    )
    quick_re = re.compile(r"^quick/([^:]+):\s*([0-9.eE+-]+)")
    recall_re = re.compile(r"^recall_(roi|rcnn)_([0-9.]+):\s*([0-9.eE+-]+)")
    class_header_re = re.compile(r"^(Car|Pedestrian|Cyclist)\s+(AP(?:_R40)?@[^:]+):$")
    metric_re = re.compile(r"^(bbox|bev|3d|aos)\s+AP:\s*([0-9.eE+-]+)")

    train_curve: List[Dict[str, Any]] = []
    quick_metrics: Dict[str, float] = {}
    recall_metrics: Dict[str, float] = {}
    official_rows: List[Dict[str, Any]] = []
    current_class: Optional[str] = None
    current_protocol: Optional[str] = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = train_re.search(line)
        if m:
            train_curve.append(
                {
                    "epoch": int(m.group(1)),
                    "epoch_total": int(m.group(2)),
                    "iter": int(m.group(3)),
                    "iter_total": int(m.group(4)),
                    "loss_batch": safe_float(m.group(5)),
                    "loss_avg": safe_float(m.group(6)),
                    "lr": safe_float(m.group(7)),
                    "acc_iter": int(m.group(8)),
                }
            )
            continue

        m = quick_re.search(line.strip())
        if m:
            quick_metrics[m.group(1)] = safe_float(m.group(2))
            continue

        m = recall_re.search(line.strip())
        if m:
            recall_metrics[f"{m.group(1)}_{m.group(2)}"] = safe_float(m.group(3))
            continue

        m = class_header_re.search(line.strip())
        if m:
            current_class = m.group(1)
            current_protocol = m.group(2)
            continue

        m = metric_re.search(line.strip())
        if m and current_class and current_protocol:
            official_rows.append(
                {
                    "class": current_class,
                    "protocol": current_protocol,
                    "metric": m.group(1),
                    "value": safe_float(m.group(2)),
                }
            )

    official_summary = summarize_official_rows(official_rows)

    return {
        "available": True,
        "path": str(path),
        "train_curve": train_curve,
        "quick_metrics": quick_metrics,
        "official_rows": official_rows,
        "official_summary": official_summary,
        "recall": recall_metrics,
        "latest_train": train_curve[-1] if train_curve else {},
    }


def summarize_official_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    classes = sorted({r.get("class") for r in rows if r.get("class")})
    out: Dict[str, Any] = {}
    selected_values: List[float] = []

    for cls in classes:
        cls_rows = [r for r in rows if r.get("class") == cls and r.get("metric") == "3d"]
        preferred = next((r for r in cls_rows if "AP_R40" in str(r.get("protocol", ""))), None)
        if preferred is None:
            preferred = cls_rows[0] if cls_rows else None
        if preferred is None:
            continue
        val = safe_float(preferred.get("value"))
        out[f"{cls}_3d_primary"] = val
        out[f"{cls}_3d_protocol"] = preferred.get("protocol", "")
        selected_values.append(val)

    if selected_values:
        out["mean_3d_primary"] = sum(selected_values) / len(selected_values)

    return out


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        out = {}
        for k, v in row.items():
            if isinstance(v, str) and v.strip() == "":
                out[k] = ""
                continue
            if isinstance(v, str):
                try:
                    if "." in v or "e" in v.lower():
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                    continue
                except Exception:
                    pass
            out[k] = v
        parsed.append(out)
    return parsed


def parse_ablation_summary(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {
            "available": False,
            "path": str(path),
            "rows": [],
            "groups": {},
            "best": {},
        }

    groups: Dict[str, List[Dict[str, Any]]] = {
        "A0_A4": [],
        "feature": [],
        "velocity": [],
        "other": [],
    }
    for row in rows:
        exp_id = str(row.get("id", ""))
        if re.match(r"^A[0-9]+$", exp_id):
            groups["A0_A4"].append(row)
        elif exp_id.startswith("feat_"):
            groups["feature"].append(row)
        elif exp_id.startswith("vel_"):
            groups["velocity"].append(row)
        else:
            groups["other"].append(row)

    best = max(rows, key=lambda r: safe_float(r.get("quick/mean_f1", -1.0), -1.0))

    return {
        "available": True,
        "path": str(path),
        "rows": rows,
        "groups": groups,
        "best": best,
    }


def parse_decode_tuning(json_path: Path, csv_path: Optional[Path] = None) -> Dict[str, Any]:
    payload = {}
    if json_path.exists() and json_path.is_file():
        payload = load_json(json_path, {})

    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    best = payload.get("best", {}) if isinstance(payload, dict) else {}

    if (not rows) and csv_path and csv_path.exists() and csv_path.is_file():
        rows = read_csv_rows(csv_path)
        if rows:
            best = max(rows, key=lambda r: safe_float(r.get("quick/mean_f1", -1.0), -1.0))

    return {
        "available": bool(rows),
        "json_path": str(json_path),
        "csv_path": str(csv_path) if csv_path else "",
        "rows": rows,
        "best": best,
    }


def deep_get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_config_summary(config_path: Optional[Path], family: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_yaml_or_json(config_path) if (config_path is not None and config_path.exists() and config_path.is_file()) else {}
    fallback = fallback or {}

    if "MODEL" in cfg:
        vfe_name = str(deep_get(cfg, ["MODEL", "VFE", "NAME"], ""))
        post_nms = str(deep_get(cfg, ["MODEL", "DENSE_HEAD", "POST_PROCESSING", "NMS_CONFIG", "NMS_TYPE"], ""))
        return {
            "input_representation": "radar7pillar" if "Radar7" in vfe_name else "dynpillar",
            "backbone": str(deep_get(cfg, ["MODEL", "BACKBONE_2D", "NAME"], "")),
            "head": str(deep_get(cfg, ["MODEL", "DENSE_HEAD", "NAME"], "CenterHead")),
            "motion_branch": "Radar7" in vfe_name,
            "vel_supervision": bool(deep_get(cfg, ["DATA_CONFIG", "APPEND_VELOCITY_TO_GT_BOXES"], False)),
            "decode_nms": post_nms,
            "evaluator": str(deep_get(cfg, ["MODEL", "POST_PROCESSING", "EVAL_METRIC"], "kitti")),
            "openpcdet": True,
        }

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    decode_cfg = cfg.get("decode", {}) if isinstance(cfg, dict) else {}

    bev_set = dataset_cfg.get("bev_feature_set", "baseline7" if family == "baseline" else "extended16")
    return {
        "input_representation": bev_set,
        "backbone": f"custom_bev_{model_cfg.get('base_channels', model_cfg.get('in_channels', 'n/a'))}",
        "head": "center_based",
        "motion_branch": bool(model_cfg.get("use_motion_branch", False)),
        "vel_supervision": bool(train_cfg.get("use_velocity_loss", False)),
        "decode_nms": str(decode_cfg.get("nms_type", "none")),
        "evaluator": "quick_f1",
        "openpcdet": False,
    }


def build_config_diff(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "input_representation",
        "backbone",
        "head",
        "motion_branch",
        "vel_supervision",
        "decode_nms",
        "evaluator",
        "openpcdet",
    ]
    rows = []
    for key in keys:
        values = {exp_id: summary.get(key) for exp_id, summary in summaries.items()}
        rows.append(
            {
                "key": key,
                "values": values,
                "different": len(set(map(str, values.values()))) > 1,
            }
        )
    return {"rows": rows}
