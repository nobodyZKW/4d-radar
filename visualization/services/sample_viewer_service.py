from __future__ import annotations

from typing import Any, Dict, List


class SampleViewerService:
    def __init__(self, registry_service, experiment_service, sample_provider):
        self.registry = registry_service
        self.experiment_service = experiment_service
        self.provider = sample_provider

    def list_samples(self, split: str, limit: int, offset: int) -> Dict[str, Any]:
        return self.provider.list_samples(split=split, limit=limit, offset=offset)

    def get_sample(self, sample_id: str, max_points: int = 8000) -> Dict[str, Any]:
        sample = self.provider.get_sample(sample_id=sample_id, max_points=max_points)
        active_exps = [
            x["id"]
            for x in self.registry.list_experiments()
            if x.get("status") in {"active", "draft"}
        ]
        return {
            "ok": True,
            "message": "",
            "sample": sample,
            "available_experiments": active_exps,
        }

    def get_predictions(self, sample_id: str, exp_ids: List[str]) -> Dict[str, Any]:
        if not exp_ids:
            exp_ids = [
                x["id"]
                for x in self.registry.list_experiments()
                if x.get("status") in {"active", "draft"}
            ]
        pred = self.provider.get_predictions(sample_id=sample_id, exp_ids=exp_ids)
        return {
            "ok": True,
            "message": "",
            **pred,
        }
