from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from data_adapters.common_metrics import load_yaml_or_json


class RegistryService:
    ABS_FIELD_MAP = {
        "history_path": "history_abs",
        "latest_metrics_path": "latest_metrics_abs",
        "config_snapshot_path": "config_snapshot_abs",
        "official_eval_path": "official_eval_abs",
        "ablation_summary_path": "ablation_summary_abs",
        "decode_tuning_path": "decode_tuning_abs",
        "prediction_dir": "prediction_abs",
    }

    def __init__(self, vis_root: Path):
        self.vis_root = vis_root
        self.config_dir = self.vis_root / "config"
        self.app_config_path = self.config_dir / "app_config.yaml"
        self.registry_path = self.config_dir / "experiments_registry.yaml"

        self.app_config = load_yaml_or_json(self.app_config_path)
        self.registry_raw = load_yaml_or_json(self.registry_path)

        self.project_root = (self.vis_root / self.app_config.get("paths", {}).get("project_root", "..")).resolve()
        self.data_root = (self.vis_root / self.app_config.get("paths", {}).get("data_root", "../../vod-min")).resolve()
        self.web_root = (self.vis_root / self.app_config.get("paths", {}).get("web_root", "web")).resolve()
        self.artifact_root = (self.vis_root / self.app_config.get("paths", {}).get("artifact_root", "artifacts")).resolve()

        self.experiments = self._normalize_experiments(self.registry_raw.get("experiments", []))

    def _to_abs(self, path_value: str) -> str:
        if not path_value:
            return ""
        p = Path(path_value)
        if p.is_absolute():
            return str(p)
        return str((self.project_root / p).resolve())

    def _normalize_experiments(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            for field, abs_field in self.ABS_FIELD_MAP.items():
                item[abs_field] = self._to_abs(str(item.get(field, "")))
            out[item["id"]] = item
        return out

    def get_app_info(self) -> Dict[str, Any]:
        return {
            "title": self.app_config.get("app", {}).get("title", "4D Radar Dashboard"),
            "default_split": self.app_config.get("app", {}).get("default_split", "val"),
            "data_root": str(self.data_root),
            "project_root": str(self.project_root),
            "artifact_root": str(self.artifact_root),
            "sample_viewer": self.app_config.get("sample_viewer", {}),
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        return sorted(self.experiments.values(), key=lambda x: (int(x.get("sort_order", 999)), x.get("id", "")))

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self.experiments.get(exp_id)

    def get_experiment_map(self) -> Dict[str, Dict[str, Any]]:
        return self.experiments

    def list_families(self) -> List[str]:
        return sorted({str(x.get("family", "")) for x in self.experiments.values() if x.get("family")})
