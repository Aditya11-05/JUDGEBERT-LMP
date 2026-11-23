"""Strict checks on backend responses. The frontend never trusts a payload just because it parsed as JSON.

Written against the real FastAPI schemas (app/schemas.py) but deliberately independent of them: the frontend talks
to the backend over HTTP only and imports nothing from `app`.
"""
from __future__ import annotations
import math

CATEGORIES = ("Preserved", "Partially Preserved", "Not Preserved")
STRENGTHS = ("strong", "moderate", "weak")


class InvalidResponse(ValueError):
    """The backend answered, but the body does not have the documented shape."""


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _dict(d, where):
    if not isinstance(d, dict):
        raise InvalidResponse(f"{where}: expected an object")
    return d


def _need(d: dict, key: str, typ, where: str):
    if key not in d or not isinstance(d[key], typ) or (typ is not bool and isinstance(d[key], bool) and typ in (int, float)):
        raise InvalidResponse(f"{where}: missing or invalid '{key}'")
    return d[key]


def _span(s, where):
    if s is None:
        return
    _dict(s, where)
    _need(s, "text", str, where), _need(s, "start", int, where), _need(s, "end", int, where)


def _valid_score(x) -> bool:
    return _is_num(x) and 1.0 <= x <= 10.0


def check_predict(d) -> dict:
    d = _dict(d, "predict response")
    status = d.get("lmp_status")
    if status not in ("available", "unavailable"):
        raise InvalidResponse("predict response: invalid 'lmp_status'")
    score, cat = d.get("predicted_lmp_score"), d.get("preservation_category")
    if status == "available":
        if not _valid_score(score) or cat not in CATEGORIES:
            raise InvalidResponse("predict response: 'available' without a valid score and category")
    elif score is not None or cat is not None:  # a score next to 'unavailable' is never displayed
        raise InvalidResponse("predict response: score present although the model is unavailable")
    findings = _need(d, "legal_error_findings", list, "predict response")
    for i, f in enumerate(findings):
        w = f"finding {i}"
        _dict(f, w)
        for k in ("error_type", "change_kind", "explanation"):
            _need(f, k, str, w)
        if f.get("evidence_strength") not in STRENGTHS:
            raise InvalidResponse(f"{w}: invalid 'evidence_strength'")
        _span(f.get("original_span"), w), _span(f.get("simplified_span"), w)
    es = _dict(d.get("explanation_status"), "explanation_status")
    _need(es, "available", bool, "explanation_status")
    bs = _dict(d.get("baseline_scores"), "baseline_scores")
    _need(bs, "requested", bool, "baseline_scores")
    for name, b in _need(bs, "scores", dict, "baseline_scores").items():
        _dict(b, f"baseline '{name}'")
        if b.get("status") == "available":
            if not _valid_score(b.get("score")):
                raise InvalidResponse(f"baseline '{name}': available without a valid score")
        elif b.get("status") == "unavailable":
            if b.get("score") is not None:
                raise InvalidResponse(f"baseline '{name}': score present although unavailable")
        else:
            raise InvalidResponse(f"baseline '{name}': invalid status")
    _need(_dict(d.get("model"), "model"), "available", bool, "model")
    for k in ("warnings", "limitations"):
        if not all(isinstance(x, str) for x in _need(d, k, list, "predict response")):
            raise InvalidResponse(f"predict response: '{k}' must be a list of strings")
    return d


def check_batch(d, expected_count: int) -> dict:
    d = _dict(d, "batch response")
    results = _need(d, "results", list, "batch response")
    if len(results) != expected_count:
        raise InvalidResponse(f"batch response: {len(results)} results for {expected_count} items")
    for i, r in enumerate(results):
        _dict(r, f"result {i}")
        if r.get("index") != i or r.get("status") not in ("ok", "error"):
            raise InvalidResponse(f"result {i}: invalid index or status")
        if r["status"] == "ok":
            check_predict(r.get("result"))
        else:
            e = _dict(r.get("error"), f"result {i} error")
            _need(e, "code", str, f"result {i} error"), _need(e, "message", str, f"result {i} error")
    return d


def check_health(d) -> dict:
    d = _dict(d, "health response")
    if d.get("status") != "ok":
        raise InvalidResponse("health response: unexpected status")
    _need(d, "ready_for_lmp_prediction", bool, "health response")
    _need(_dict(d.get("model"), "health.model"), "available", bool, "health.model")
    ex = _dict(d.get("explanations"), "health.explanations")
    _need(ex, "available", bool, "health.explanations"), _need(ex, "supported_languages", list, "health.explanations")
    for name, b in _need(d, "baselines", dict, "health response").items():
        _need(_dict(b, f"baseline '{name}'"), "available", bool, f"baseline '{name}'")
    return d


def check_model_info(d) -> dict:
    d = _dict(d, "model-info response")
    _need(d, "backbone", str, "model-info"), _need(d, "supported_languages", list, "model-info")
    _need(_dict(d.get("model"), "model-info.model"), "available", bool, "model-info.model")
    _need(d, "planned_languages_not_supported", list, "model-info")
    ev = _dict(d.get("evaluation"), "model-info.evaluation")
    if ev.get("status") not in ("saved_results", "no_saved_results", "unreadable"):
        raise InvalidResponse("model-info.evaluation: invalid status")
    _need(d, "baselines", dict, "model-info")
    return d
