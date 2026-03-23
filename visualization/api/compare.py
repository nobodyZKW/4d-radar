from __future__ import annotations

from typing import Any, Dict, List


def compare_api(compare_service, query: Dict[str, Any]) -> Dict[str, Any]:
    ids_str = str(query.get("ids", [""])[0]).strip()
    ids: List[str] = [x.strip() for x in ids_str.split(",") if x.strip()] if ids_str else []
    return compare_service.compare(ids)
