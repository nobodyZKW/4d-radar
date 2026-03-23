from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common_metrics import (
    extract_config_summary,
    parse_ablation_summary,
    parse_decode_tuning,
    parse_openpcdet_train_log,
    safe_float,
    to_optional_path,
)


class ImproveV15Adapter:
    family = "improve-v1.5"

    def _resolve_log_path(self, exp: Dict[str, Any]) -> Path | None:
        history_path = to_optional_path(exp.get("history_abs"))
        if history_path is None:
            return None
        if history_path.is_file():
            return history_path

        parent = history_path if history_path.is_dir() else history_path.parent
        if not parent.exists():
            return history_path
        logs = sorted(parent.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
        return logs[-1] if logs else history_path

    def _ablation_entry_from_id(self, exp_id: str) -> str:
        mapping = {
            "improve_v1_5_A0": "A0",
            "improve_v1_5_A1": "A1",
            "improve_v1_5_A2": "A2",
            "improve_v1_5_A3": "A3",
            "improve_v1_5_A4": "A4",
        }
        return mapping.get(exp_id, "")

    def _pick_quick_summary(
        self,
        exp: Dict[str, Any],
        quick_metrics: Dict[str, float],
        ablation: Dict[str, Any],
    ) -> Dict[str, Any]:
        exp_id = exp.get("id", "")
        rows: List[Dict[str, Any]] = ablation.get("rows", []) if isinstance(ablation, dict) else []

        if exp_id in {"improve_v1_5_feat_ablation", "improve_v1_5_vel_ablation"}:
            return {
                "mean_f1": safe_float(ablation.get("best", {}).get("quick/mean_f1", 0.0)),
                "source": "ablation_summary",
            }

        ablation_id = self._ablation_entry_from_id(exp_id)
        if ablation_id and rows:
            row = next((r for r in rows if str(r.get("id", "")) == ablation_id), None)
            if row:
                return {
                    "mean_f1": safe_float(row.get("quick/mean_f1", 0.0)),
                    "source": "ablation_row",
                }

        if exp_id == "improve_v1_5_main":
            mean_f1 = safe_float(quick_metrics.get("mean_f1", 0.0))
            if mean_f1 <= 0.0 and rows:
                row = next((r for r in rows if str(r.get("id", "")) == "A4"), None)
                if row:
                    return {
                        "mean_f1": safe_float(row.get("quick/mean_f1", 0.0)),
                        "source": "ablation_A4_fallback",
                    }

        return {
            "mean_f1": safe_float(quick_metrics.get("mean_f1", 0.0)),
            "Car_f1": safe_float(quick_metrics.get("Car_f1", 0.0)),
            "Pedestrian_f1": safe_float(quick_metrics.get("Pedestrian_f1", 0.0)),
            "Cyclist_f1": safe_float(quick_metrics.get("Cyclist_f1", 0.0)),
            "source": "openpcdet_log",
        }

    def load(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        log_path = self._resolve_log_path(exp)
        log_data = (
            parse_openpcdet_train_log(log_path)
            if log_path is not None
            else {
                "available": False,
                "path": "",
                "train_curve": [],
                "quick_metrics": {},
                "official_rows": [],
                "official_summary": {},
                "recall": {},
                "latest_train": {},
            }
        )

        ablation_path = to_optional_path(exp.get("ablation_summary_abs"))
        ablation_data = parse_ablation_summary(ablation_path) if ablation_path is not None else {
            "available": False,
            "rows": [],
            "groups": {},
            "best": {},
            "path": "",
        }

        decode_json = to_optional_path(exp.get("decode_tuning_abs"))
        decode_csv = decode_json.with_suffix(".csv") if decode_json is not None else None
        decode_data = parse_decode_tuning(decode_json, decode_csv) if decode_json is not None else {
            "available": False,
            "rows": [],
            "best": {},
            "json_path": "",
            "csv_path": "",
        }

        config_path = to_optional_path(exp.get("config_snapshot_abs"))
        official_eval_path = to_optional_path(exp.get("official_eval_abs"))

        quick_summary = self._pick_quick_summary(exp, log_data.get("quick_metrics", {}), ablation_data)

        official_summary = log_data.get("official_summary", {})
        official_available = bool(log_data.get("official_rows"))

        # For experiments that are pure ablation views without dedicated logs,
        # keep quick metrics from ablation and mark official as missing.
        if exp.get("quick_metrics_type") in {"ablation_row", "ablation_table"}:
            official_available = False

        quick_available = bool(log_data.get("quick_metrics")) or bool(ablation_data.get("rows"))

        detail_payload = {
            "id": exp["id"],
            "family": exp["family"],
            "display_name": exp.get("display_name", exp["id"]),
            "status": exp.get("status", "active"),
            "description": exp.get("description", ""),
            "tags": exp.get("tags", []),
            "color": exp.get("color", "#ff6b35"),
            "sort_order": exp.get("sort_order", 999),
            "quick_metrics": {
                "available": quick_available,
                "type": exp.get("quick_metrics_type", "openpcdet_log"),
                "path": str(log_path) if log_path else "",
                "summary": quick_summary,
                "latest": log_data.get("latest_train", {}),
                "quick_values": log_data.get("quick_metrics", {}),
                "curves": {
                    "acc_iter": [x.get("acc_iter") for x in log_data.get("train_curve", [])],
                    "loss_avg": [x.get("loss_avg") for x in log_data.get("train_curve", [])],
                    "lr": [x.get("lr") for x in log_data.get("train_curve", [])],
                },
            },
            "official_metrics": {
                "available": official_available,
                "type": exp.get("official_metrics_type", "kitti_eval_log"),
                "summary": official_summary,
                "rows": log_data.get("official_rows", []),
                "recall": log_data.get("recall", {}),
                "message": "" if official_available else "该实验暂缺可解析的 official 指标。",
            },
            "ablation": ablation_data,
            "decode_tuning": decode_data,
            "config_summary": extract_config_summary(config_path, family=exp["family"], fallback=exp),
            "artifacts": {
                "history_path": str(log_path) if log_path else "",
                "official_eval_path": str(official_eval_path) if official_eval_path else "",
                "ablation_summary_path": str(ablation_path) if ablation_path else "",
                "decode_tuning_path": str(decode_json) if decode_json else "",
                "config_snapshot_path": str(config_path) if config_path else "",
                "prediction_dir": exp.get("prediction_abs", ""),
            },
        }

        # Specialized summaries for feature / velocity views.
        if exp.get("id") == "improve_v1_5_feat_ablation":
            detail_payload["ablation_focus"] = {
                "group": "feature",
                "rows": ablation_data.get("groups", {}).get("feature", []),
            }
        elif exp.get("id") == "improve_v1_5_vel_ablation":
            detail_payload["ablation_focus"] = {
                "group": "velocity",
                "rows": ablation_data.get("groups", {}).get("velocity", []),
            }

        return detail_payload
