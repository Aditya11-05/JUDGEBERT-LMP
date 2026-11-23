"""Builders for backend-shaped payloads used by frontend tests. They are FIXTURES, not real model output.
test_frontend_client.py checks them against the real backend schemas so they cannot drift from the API."""

LIMITS = ["Research prototype; not legal advice.", "Rule-based findings never influence the LMP score."]


def finding(error_type="obligation_permission_change", strength="strong", o=("must", 11, 15), s=("may", 11, 14), kind="changed"):
    sp = lambda t: None if t is None else {"text": t[0], "start": t[1], "end": t[2]}
    return {"error_type": error_type, "change_kind": kind, "original_span": sp(o), "simplified_span": sp(s),
            "explanation": "Potential meaning change.", "evidence_strength": strength, "rule_id": "modality.changed"}


def predict_response(available=False, score=None, category=None, findings=None, baselines=None, reason="weights_missing",
                     requested=False, explanations=True):
    if available:
        reason = None
    return {
        "language": "en", "lmp_status": "available" if available else "unavailable",
        "predicted_lmp_score": score if available else None, "preservation_category": category if available else None,
        "lmp_unavailable_reason": reason, "legal_error_findings": [finding()] if findings is None else findings,
        "highlighted_original": None, "highlighted_simplified": None, "explanation_summary": "1 potential change flagged.",
        "explanation_status": {"available": explanations, "reason": None, "message": None},
        "baseline_scores": {"requested": requested, "scores": baselines or {}},
        "model": {"available": available, "backbone": "FacebookAI/xlm-roberta-base", "version": "fixture-v0" if available else None,
                  "reason": reason, "message": "No trained LMP model is installed; LMP scores are not available." if not available else None},
        "model_version": "fixture-v0" if available else None, "warnings": [] if available else ["No LMP score was produced."],
        "limitations": LIMITS}


def health(ready=False, baselines=None):
    return {"status": "ok", "ready_for_lmp_prediction": ready,
            "model": {"available": ready, "backbone": "FacebookAI/xlm-roberta-base", "version": "v" if ready else None,
                      "reason": None if ready else "weights_missing", "message": None if ready else "No trained LMP model is installed."},
            "explanations": {"available": True, "supported_languages": ["en"], "reason": None},
            "baselines": baselines or {"tfidf": {"available": False, "reason": "artifact_missing", "message": "No fitted TF-IDF baseline artifact is installed."},
                                       "multilingual_embedding": {"available": False, "reason": "not_implemented", "message": "Not implemented yet."}}}


def model_info(ready=False, evaluation=None):
    return {"backbone": "FacebookAI/xlm-roberta-base",
            "model": health(ready)["model"], "supported_languages": ["en"], "planned_languages_not_supported": ["hi"],
            "score_range": [1.0, 10.0], "category_thresholds": {"preserved_min": 7.0, "partial_min": 4.0},
            "baselines": {"tfidf": {"available": False, "reason": "artifact_missing", "message": "none", "metadata": None},
                          "multilingual_embedding": {"available": False, "reason": "not_implemented", "message": "none", "metadata": None}},
            "evaluation": evaluation or {"status": "no_saved_results", "source": None, "metrics": None}, "disclaimer": "Research prototype."}


def batch_response(items, available=False, score=8.0, category="Preserved", errors=None):
    """One result per item; `errors` = {index: (code, message)} produces per-item error entries."""
    errors, results = errors or {}, []
    for i, _ in enumerate(items):
        if i in errors:
            results.append({"index": i, "status": "error", "result": None, "error": {"code": errors[i][0], "message": errors[i][1]}})
        else:
            results.append({"index": i, "status": "ok", "error": None,
                            "result": predict_response(available, score, category)})
    ok = len(results) - len(errors)
    return {"count": len(results), "ok_count": ok, "error_count": len(errors),
            "lmp_status": "available" if available else "unavailable", "results": results}
