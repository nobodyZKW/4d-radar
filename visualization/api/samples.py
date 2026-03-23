from __future__ import annotations

from typing import Any, Dict


def _parse_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except Exception:
        return default
    return max(min_value, min(max_value, n))


def list_samples_api(sample_service, query: Dict[str, Any]) -> Dict[str, Any]:
    split = str(query.get("split", ["val"])[0]).strip()
    limit = _parse_int(str(query.get("limit", ["300"])[0]), 300, 1, 50000)
    offset = _parse_int(str(query.get("offset", ["0"])[0]), 0, 0, 1000000)
    return sample_service.list_samples(split=split, limit=limit, offset=offset)


def sample_detail_api(sample_service, sample_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    max_points = _parse_int(str(query.get("max_points", ["8000"])[0]), 8000, 100, 50000)
    return sample_service.get_sample(sample_id=sample_id, max_points=max_points)


def sample_predictions_api(sample_service, sample_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    exp_param = str(query.get("exp", [""])[0]).strip()
    exp_ids = [x.strip() for x in exp_param.split(",") if x.strip()] if exp_param else []
    return sample_service.get_predictions(sample_id=sample_id, exp_ids=exp_ids)
