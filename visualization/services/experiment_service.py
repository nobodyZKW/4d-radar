from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_adapters import BaselineAdapter, ImproveV1Adapter, ImproveV15Adapter


class ExperimentService:
    def __init__(self, registry_service):
        self.registry = registry_service
        self.adapters = {
            "baseline": BaselineAdapter(),
            "improve-v1": ImproveV1Adapter(),
            "improve-v1.5": ImproveV15Adapter(),
        }

    def _pick_adapter(self, family: str):
        return self.adapters.get(family)

    def _mtime_iso(self, paths: List[str]) -> str:
        mtimes = []
        for p in paths:
            if not p:
                continue
            path = Path(p)
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        if not mtimes:
            return ""
        return datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")

    def get_experiment_detail(self, exp_id: str) -> Dict[str, Any]:
        exp = self.registry.get_experiment(exp_id)
        if not exp:
            return {"ok": False, "message": f"experiment not found: {exp_id}", "experiment": {}}

        adapter = self._pick_adapter(exp.get("family", ""))
        if adapter is None:
            return {"ok": False, "message": f"unsupported family: {exp.get('family')}", "experiment": {}}

        detail = adapter.load(exp)
        artifact_paths = detail.get("artifacts", {})
        detail["updated_at"] = self._mtime_iso([str(v) for v in artifact_paths.values()])
        detail["registry"] = exp
        return {"ok": True, "message": "", "experiment": detail}

    def get_experiment_metrics(self, exp_id: str) -> Dict[str, Any]:
        detail = self.get_experiment_detail(exp_id)
        if not detail.get("ok"):
            return detail
        exp = detail["experiment"]
        return {
            "ok": True,
            "message": "",
            "id": exp_id,
            "quick_metrics": exp.get("quick_metrics", {}),
            "official_metrics": exp.get("official_metrics", {}),
            "ablation": exp.get("ablation", {}),
            "decode_tuning": exp.get("decode_tuning", {}),
        }

    def list_experiments(
        self,
        family: str = "",
        status: str = "",
        tags: Optional[List[str]] = None,
        sort: str = "sort_order",
    ) -> Dict[str, Any]:
        tags = tags or []
        rows = []
        for exp in self.registry.list_experiments():
            if family and exp.get("family") != family:
                continue
            if status and exp.get("status") != status:
                continue
            exp_tags = set(exp.get("tags", []))
            if tags and not set(tags).issubset(exp_tags):
                continue

            detail = self.get_experiment_detail(exp["id"])
            if not detail.get("ok"):
                row = {
                    "id": exp["id"],
                    "display_name": exp.get("display_name", exp["id"]),
                    "family": exp.get("family", ""),
                    "status": exp.get("status", ""),
                    "description": exp.get("description", ""),
                    "quick_summary": {},
                    "official_summary": {},
                    "updated_at": "",
                    "tags": exp.get("tags", []),
                    "sort_order": exp.get("sort_order", 999),
                    "error": detail.get("message", ""),
                }
            else:
                info = detail["experiment"]
                row = {
                    "id": info.get("id"),
                    "display_name": info.get("display_name"),
                    "family": info.get("family"),
                    "status": info.get("status"),
                    "description": info.get("description"),
                    "quick_summary": info.get("quick_metrics", {}).get("summary", {}),
                    "official_summary": info.get("official_metrics", {}).get("summary", {}),
                    "official_available": info.get("official_metrics", {}).get("available", False),
                    "quick_available": info.get("quick_metrics", {}).get("available", False),
                    "updated_at": info.get("updated_at", ""),
                    "tags": info.get("tags", []),
                    "sort_order": info.get("sort_order", 999),
                }
            rows.append(row)

        if sort == "metric":
            rows.sort(key=lambda x: float(x.get("quick_summary", {}).get("mean_f1", -1.0) or -1.0), reverse=True)
        elif sort == "time":
            rows.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        elif sort == "family":
            rows.sort(key=lambda x: (x.get("family", ""), int(x.get("sort_order", 999))))
        else:
            rows.sort(key=lambda x: int(x.get("sort_order", 999)))

        families = self.registry.list_families()
        return {
            "ok": True,
            "message": "",
            "families": families,
            "count": len(rows),
            "experiments": rows,
        }

    def get_family_overview(self) -> Dict[str, Any]:
        exps = self.list_experiments().get("experiments", [])
        fam_map: Dict[str, List[Dict[str, Any]]] = {}
        for row in exps:
            fam_map.setdefault(row.get("family", "unknown"), []).append(row)

        families = []
        for family, items in sorted(fam_map.items(), key=lambda x: x[0]):
            main = next((x for x in items if "main" in x.get("tags", [])), items[0]) if items else None
            families.append(
                {
                    "family": family,
                    "count": len(items),
                    "main": main,
                    "items": items,
                }
            )

        return {
            "ok": True,
            "message": "",
            "families": families,
        }
