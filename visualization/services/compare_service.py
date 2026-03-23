from __future__ import annotations

from typing import Any, Dict, List

from data_adapters.common_metrics import build_config_diff


class CompareService:
    def __init__(self, experiment_service):
        self.experiment_service = experiment_service

    def compare(self, ids: List[str]) -> Dict[str, Any]:
        ids = [x for x in ids if x]
        details = []
        for exp_id in ids:
            data = self.experiment_service.get_experiment_detail(exp_id)
            if data.get("ok"):
                details.append(data["experiment"])

        quick_rows = []
        official_rows = []
        config_summaries = {}

        for exp in details:
            quick = exp.get("quick_metrics", {}).get("summary", {})
            official = exp.get("official_metrics", {}).get("summary", {})
            quick_rows.append(
                {
                    "id": exp.get("id"),
                    "display_name": exp.get("display_name"),
                    "family": exp.get("family"),
                    "mean_f1": quick.get("mean_f1", quick.get("mean_f1", 0.0)),
                    "Car_f1": quick.get("Car_f1", 0.0),
                    "Pedestrian_f1": quick.get("Pedestrian_f1", 0.0),
                    "Cyclist_f1": quick.get("Cyclist_f1", 0.0),
                    "loss": quick.get("loss", 0.0),
                    "val_loss": quick.get("val_loss", 0.0),
                }
            )
            official_rows.append(
                {
                    "id": exp.get("id"),
                    "display_name": exp.get("display_name"),
                    "official_available": exp.get("official_metrics", {}).get("available", False),
                    "mean_3d_primary": official.get("mean_3d_primary", None),
                    "Car_3d_primary": official.get("Car_3d_primary", None),
                    "Pedestrian_3d_primary": official.get("Pedestrian_3d_primary", None),
                    "Cyclist_3d_primary": official.get("Cyclist_3d_primary", None),
                }
            )
            config_summaries[exp.get("id")] = exp.get("config_summary", {})

        config_diff = build_config_diff(config_summaries) if config_summaries else {"rows": []}

        curve_pack = {}
        for exp in details:
            curves = exp.get("quick_metrics", {}).get("curves", {})
            if curves:
                curve_pack[exp.get("id")] = curves

        return {
            "ok": True,
            "message": "",
            "ids": ids,
            "experiments": details,
            "quick_table": quick_rows,
            "official_table": official_rows,
            "config_diff": config_diff,
            "curves": curve_pack,
        }
