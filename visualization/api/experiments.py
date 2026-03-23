from __future__ import annotations

from typing import Any, Dict


def list_experiments_api(experiment_service, query: Dict[str, Any]) -> Dict[str, Any]:
    family = str(query.get("family", [""])[0]).strip()
    status = str(query.get("status", [""])[0]).strip()
    tags_str = str(query.get("tags", [""])[0]).strip()
    sort = str(query.get("sort", ["sort_order"])[0]).strip()
    tags = [x.strip() for x in tags_str.split(",") if x.strip()] if tags_str else []
    return experiment_service.list_experiments(family=family, status=status, tags=tags, sort=sort)


def experiment_detail_api(experiment_service, exp_id: str) -> Dict[str, Any]:
    return experiment_service.get_experiment_detail(exp_id)


def experiment_metrics_api(experiment_service, exp_id: str) -> Dict[str, Any]:
    return experiment_service.get_experiment_metrics(exp_id)
