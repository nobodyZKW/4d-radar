from __future__ import annotations

from typing import Any, Dict

from .common_metrics import extract_config_summary, load_history_json, safe_float, to_optional_path


class ImproveV1Adapter:
    family = "improve-v1"

    def load(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        history_path = to_optional_path(exp.get("history_abs"))
        config_path = to_optional_path(exp.get("config_snapshot_abs"))

        history_data = (
            load_history_json(history_path)
            if history_path is not None
            else {"available": False, "path": "", "latest": {}, "best": {}, "curves": {}}
        )
        latest = history_data.get("latest", {})

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
            "color": exp.get("color", "#2a9d8f"),
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
                "message": "improve-v1 当前以 quick 指标为主，official 指标暂缺。",
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
                "config_snapshot_path": str(config_path) if config_path else "",
                "prediction_dir": exp.get("prediction_abs", ""),
            },
        }
