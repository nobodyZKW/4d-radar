from __future__ import annotations

from typing import Any, Dict

from .common_metrics import (
    extract_config_summary,
    load_history_json,
    load_json,
    safe_float,
    to_optional_path,
)


class BaselineAdapter:
    family = "baseline"

    def load(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        history_path = to_optional_path(exp.get("history_abs"))
        latest_metrics_path = to_optional_path(exp.get("latest_metrics_abs"))
        config_path = to_optional_path(exp.get("config_snapshot_abs"))

        history_data = (
            load_history_json(history_path)
            if history_path is not None
            else {"available": False, "path": "", "latest": {}, "best": {}, "curves": {}}
        )
        latest_metrics = (
            load_json(latest_metrics_path, {}) if latest_metrics_path is not None and latest_metrics_path.exists() else {}
        )
        latest = dict(history_data.get("latest", {}))
        latest.update({k: v for k, v in latest_metrics.items() if k not in latest})

        quick_summary = {
            "mean_f1": safe_float(latest.get("mean_f1", 0.0)),
            "Car_f1": safe_float(latest.get("Car_f1", 0.0)),
            "Pedestrian_f1": safe_float(latest.get("Pedestrian_f1", 0.0)),
            "Cyclist_f1": safe_float(latest.get("Cyclist_f1", 0.0)),
            "loss": safe_float(latest.get("loss_total", latest.get("loss", 0.0))),
            "val_loss": safe_float(latest.get("val_loss_total", latest.get("val_loss", 0.0))),
        }

        return {
            "id": exp["id"],
            "family": exp["family"],
            "display_name": exp.get("display_name", exp["id"]),
            "status": exp.get("status", "active"),
            "description": exp.get("description", ""),
            "tags": exp.get("tags", []),
            "color": exp.get("color", "#3a86ff"),
            "sort_order": exp.get("sort_order", 999),
            "quick_metrics": {
                "available": history_data.get("available", False),
                "type": exp.get("quick_metrics_type", "history_json"),
                "path": history_data.get("path", ""),
                "summary": quick_summary,
                "latest": latest,
                "best": history_data.get("best", {}),
                "curves": history_data.get("curves", {}),
            },
            "official_metrics": {
                "available": False,
                "type": exp.get("official_metrics_type", "none"),
                "summary": {},
                "rows": [],
                "message": "该实验暂未接入 official evaluator 结果。",
            },
            "ablation": {
                "available": False,
                "rows": [],
            },
            "decode_tuning": {
                "available": False,
                "rows": [],
            },
            "config_summary": extract_config_summary(config_path, family=exp["family"], fallback=exp),
            "artifacts": {
                "history_path": str(history_path) if history_path else "",
                "latest_metrics_path": str(latest_metrics_path) if latest_metrics_path else "",
                "config_snapshot_path": str(config_path) if config_path else "",
                "prediction_dir": exp.get("prediction_abs", ""),
            },
        }
