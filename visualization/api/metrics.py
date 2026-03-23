from __future__ import annotations

from typing import Any, Dict


def ablations_api(experiment_service, query: Dict[str, Any]) -> Dict[str, Any]:
    family = str(query.get("family", [""])[0]).strip()
    exps = experiment_service.list_experiments(family=family).get("experiments", [])
    payload = []
    for row in exps:
        detail = experiment_service.get_experiment_detail(row.get("id", ""))
        if not detail.get("ok"):
            continue
        exp = detail["experiment"]
        abl = exp.get("ablation", {})
        if not abl.get("available"):
            continue
        payload.append(
            {
                "id": exp.get("id"),
                "display_name": exp.get("display_name"),
                "family": exp.get("family"),
                "ablation": abl,
                "focus": exp.get("ablation_focus", {}),
            }
        )
    return {"ok": True, "message": "", "rows": payload}


def decode_tuning_api(experiment_service, query: Dict[str, Any]) -> Dict[str, Any]:
    exp_id = str(query.get("exp", [""])[0]).strip()
    if not exp_id:
        exp_id = "improve_v1_5_main"
    detail = experiment_service.get_experiment_detail(exp_id)
    if not detail.get("ok"):
        return detail
    exp = detail["experiment"]
    decode = exp.get("decode_tuning", {})
    return {
        "ok": True,
        "message": "",
        "id": exp_id,
        "decode_tuning": decode,
    }
