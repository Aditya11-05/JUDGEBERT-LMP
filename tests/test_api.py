"""API tests WITHOUT any model: the 'trained model unavailable' configuration (no mocks here)."""
import json
import re

import pytest
import yaml
from app.config import ROOT, Settings
from app.explanations import SUPPORTED_LANGUAGES
from app.inference import ModelStatus
from app.schemas import (BatchResponse, ErrorResponse, HealthResponse, ModelInfoResponse, PredictResponse)
from conftest import PAIR

BAD_PAIRS = [
    ("identical", "The tenant must pay rent.", "The tenant must pay rent."),
    ("unrelated", "The tenant must pay rent.", "Gardens are lovely in spring."),
    ("heavily changed", "All tenants must pay Rs. 10,000 by the fifth day unless the landlord agrees.",
     "Some tenants may pay Rs. 1,000 by the tenth day."),
]


# ------------------------------------------------------------------ /health
def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = HealthResponse.model_validate(r.json())
    assert h.status == "ok" and h.ready_for_lmp_prediction is False
    assert h.model.available is False and h.model.reason == "weights_missing"
    assert h.explanations.available is True and h.explanations.supported_languages == ["en"]
    assert h.baselines["tfidf"].available is False and h.baselines["tfidf"].reason == "artifact_missing"
    assert h.baselines["multilingual_embedding"].reason == "not_implemented"


def test_manifest_without_weights_is_not_ready(tmp_path, make_client):
    d = tmp_path / "model"
    d.mkdir()
    (d / "lmp_manifest.json").write_text(json.dumps({"version": "v1", "backbone": "x", "weights_file": "w.bin"}))
    h = make_client().get("/health").json()
    assert h["ready_for_lmp_prediction"] is False and h["model"]["reason"] == "weights_missing"


def test_weights_present_without_tokenizer_is_not_ready(tmp_path, make_client):
    d = tmp_path / "model"
    d.mkdir()
    (d / "lmp_manifest.json").write_text(json.dumps({"version": "v1", "backbone": "x", "weights_file": "w.bin"}))
    (d / "w.bin").write_bytes(b"not a real model")
    c = make_client()
    h = c.get("/health").json()
    assert h["ready_for_lmp_prediction"] is False and h["model"]["reason"] == "weights_missing"
    assert c.post("/predict", json=PAIR).json()["predicted_lmp_score"] is None


@pytest.mark.parametrize("manifest", ["{not json", json.dumps({"version": "v"}),
                                      json.dumps({"version": "v", "backbone": "b", "weights_file": "../escape.bin"})])
def test_invalid_manifest(tmp_path, make_client, manifest):
    d = tmp_path / "model"
    d.mkdir()
    (d / "lmp_manifest.json").write_text(manifest)
    h = make_client().get("/health").json()
    assert h["ready_for_lmp_prediction"] is False and h["model"]["reason"] == "manifest_invalid"


def test_model_is_loaded_once_at_startup_not_per_request(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app.main as main
    from conftest import make_settings
    calls = []

    def counting_loader(settings):
        calls.append(1)
        return None, ModelStatus(False, "weights_missing", "none", None, settings.backbone)

    monkeypatch.setattr(main, "load_predictor", counting_loader)
    with TestClient(main.create_app(make_settings(tmp_path))) as c:
        for _ in range(5):
            c.post("/predict", json=PAIR)
            c.get("/health")
    assert len(calls) == 1


def test_loader_crash_degrades_to_unavailable(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app.main as main
    from conftest import make_settings
    monkeypatch.setattr(main, "load_predictor", lambda s: (_ for _ in ()).throw(RuntimeError("boom /secret/path")))
    with TestClient(main.create_app(make_settings(tmp_path))) as c:
        h = c.get("/health").json()
        assert h["model"]["reason"] == "load_error" and "secret" not in json.dumps(h)
        assert c.post("/predict", json=PAIR).json()["lmp_status"] == "unavailable"


# ------------------------------------------------------------------ /predict, model unavailable
def test_predict_reports_unavailable_explicitly(client):
    r = client.post("/predict", json=PAIR)
    assert r.status_code == 200
    d = PredictResponse.model_validate(r.json()).model_dump()
    assert d["lmp_status"] == "unavailable" and d["lmp_unavailable_reason"] == "weights_missing"
    assert d["predicted_lmp_score"] is None and d["preservation_category"] is None
    assert d["model_version"] is None and d["model"]["available"] is False
    assert any("No LMP score" in w for w in d["warnings"])
    assert any("not legal advice" in x.lower() for x in d["limitations"])


@pytest.mark.parametrize("label,o,s", BAD_PAIRS)
def test_no_fabricated_score_whatever_the_rules_find(client, label, o, s):
    raw = client.post("/predict", json={"original_text": o, "simplified_text": s, "include_baselines": True}).text
    d = json.loads(raw)
    assert d["predicted_lmp_score"] is None and d["preservation_category"] is None and d["lmp_status"] == "unavailable"
    # no numeric score hiding anywhere under a score-like key
    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if ("score" in k or "similarity" in k) and not isinstance(v, (dict, list)):
                    assert v is None, f"{path}/{k} = {v}"
                walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
    walk(d)


def test_predict_returns_explanations_from_existing_module(client):
    d = client.post("/predict", json=PAIR).json()
    types = {f["error_type"] for f in d["legal_error_findings"]}
    assert types == {"obligation_permission_change", "date_or_deadline_change"}
    modal = next(f for f in d["legal_error_findings"] if f["error_type"] == "obligation_permission_change")
    assert modal["original_span"]["text"] == "must" and modal["simplified_span"]["text"] == "may"
    assert "[[must]]" in d["highlighted_original"] and "[[may]]" in d["highlighted_simplified"]
    assert d["explanation_status"]["available"] is True and d["explanation_summary"]


def test_harmless_paraphrase_has_no_findings(client):
    d = client.post("/predict", json={
        "original_text": "The tenant must pay the rent by the fifth day of each month.",
        "simplified_text": "The tenant is required to pay rent no later than the fifth day of every month."}).json()
    assert d["legal_error_findings"] == [] and d["predicted_lmp_score"] is None


def test_language_normalisation(client):
    for lang in ("EN", " en ", "en-IN", "en_US"):
        r = client.post("/predict", json={**PAIR, "language": lang})
        assert r.status_code == 200 and r.json()["language"] == "en"


def test_context_and_domain_are_accepted_but_flagged_unused(client):
    d = client.post("/predict", json={**PAIR, "context": "Clause 4", "legal_domain": "rental"}).json()
    assert any("'context'" in w for w in d["warnings"]) and any("'legal_domain'" in w for w in d["warnings"])


# ------------------------------------------------------------------ languages
@pytest.mark.parametrize("lang", ["hi", "fr", "de", "hi-IN"])
def test_unsupported_languages_rejected_clearly(client, lang):
    r = client.post("/predict", json={**PAIR, "language": lang})
    assert r.status_code == 422
    ErrorResponse.model_validate(r.json())
    e = r.json()["error"]
    assert e["code"] == "unsupported_language" and e["supported_languages"] == ["en"]
    assert "legal_error_findings" not in r.json()


def test_hindi_stays_unsupported_even_though_the_yaml_lists_it_as_a_candidate(client):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    assert "hi" in cfg["languages"]  # candidate only
    assert SUPPORTED_LANGUAGES == ("en",)
    assert client.post("/predict", json={**PAIR, "language": "hi"}).status_code == 422
    assert client.get("/model-info").json()["supported_languages"] == ["en"]


@pytest.mark.parametrize("lang", ["e!", "english", "", "x", 123, None, ["en"]])
def test_malformed_language_is_a_validation_error(client, lang):
    r = client.post("/predict", json={**PAIR, "language": lang})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# ------------------------------------------------------------------ malformed / empty / long input
@pytest.mark.parametrize("body", [
    {"original_text": "", "simplified_text": "x"},
    {"original_text": "x", "simplified_text": ""},
    {"original_text": "   \n\t", "simplified_text": "x"},
    {"original_text": "x", "simplified_text": "  "},
    {"original_text": "x"},
    {"simplified_text": "x"},
    {},
    {"original_text": 5, "simplified_text": "x"},
    {"original_text": "x", "simplified_text": ["y"]},
    {"original_text": None, "simplified_text": "x"},
    {"original_text": "x", "simplified_text": "y", "include_baselines": "maybe"},
    {"original_text": "x", "simplified_text": "y", "unexpected_field": 1},
    {"original_text": "x", "simplified_text": "y", "legal_domain": "d" * 101},
])
def test_invalid_bodies_are_422_with_safe_error(client, body):
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    ErrorResponse.model_validate(r.json())
    assert r.json()["error"]["code"] == "validation_error" and r.json()["error"]["details"]


def test_validation_errors_do_not_echo_input(client):
    secret = "SUPER-SECRET-VALUE-12345"
    r = client.post("/predict", json={"original_text": secret, "simplified_text": "x", "bogus": secret})
    assert r.status_code == 422 and secret not in r.text


@pytest.mark.parametrize("kw", [{"content": "not json", "headers": {"content-type": "application/json"}},
                                {"content": "[1,2]", "headers": {"content-type": "application/json"}},
                                {"content": "null", "headers": {"content-type": "application/json"}},
                                {"content": "a=b", "headers": {"content-type": "application/x-www-form-urlencoded"}},
                                {"content": b"\xff\xfe", "headers": {"content-type": "application/json"}}])
def test_malformed_payloads(client, kw):
    r = client.post("/predict", **kw)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_text_length_limits(make_client):
    c = make_client(max_chars=50)
    ok = c.post("/predict", json={"original_text": "a " * 25, "simplified_text": "b"})
    assert ok.status_code == 200
    for field in ("original_text", "simplified_text", "context"):
        body = {"original_text": "x", "simplified_text": "y", field: "z" * 51}
        r = c.post("/predict", json=body)
        assert r.status_code == 422 and r.json()["error"]["code"] == "input_too_long"
        assert field in r.json()["error"]["message"]


def test_oversized_body_is_413(make_client):
    c = make_client(max_body_bytes=300)
    r = c.post("/predict", json={"original_text": "x" * 1000, "simplified_text": "y"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


# ------------------------------------------------------------------ baselines when artifacts are missing
def test_missing_baseline_artifacts_are_reported_not_faked(client):
    d = client.post("/predict", json={**PAIR, "include_baselines": True}).json()
    b = d["baseline_scores"]
    assert b["requested"] is True
    assert {k: v["status"] for k, v in b["scores"].items()} == {"tfidf": "unavailable", "multilingual_embedding": "unavailable"}
    assert b["scores"]["tfidf"]["reason"] == "artifact_missing" and b["scores"]["tfidf"]["score"] is None
    assert b["scores"]["multilingual_embedding"]["reason"] == "not_implemented"
    assert any("no fitted baseline" in w for w in d["warnings"])


def test_baselines_not_requested_means_no_scores(client):
    b = client.post("/predict", json=PAIR).json()["baseline_scores"]
    assert b == {"requested": False, "scores": {}}


# ------------------------------------------------------------------ batch
def test_batch_without_model(client):
    items = [PAIR, {"original_text": "Pay Rs. 500.", "simplified_text": "Pay Rs. 50."}, {"original_text": "A b.", "simplified_text": "A b."}]
    r = client.post("/batch-predict", json={"items": items})
    assert r.status_code == 200
    b = BatchResponse.model_validate(r.json())
    assert (b.count, b.ok_count, b.error_count, b.lmp_status) == (3, 3, 0, "unavailable")
    assert [x.index for x in b.results] == [0, 1, 2]
    assert all(x.result.predicted_lmp_score is None for x in b.results)
    assert b.results[1].result.legal_error_findings[0].error_type == "monetary_amount_change"
    assert b.results[2].result.legal_error_findings == []


def test_batch_reports_per_item_errors_and_keeps_good_items(make_client):
    c = make_client(max_chars=60)
    items = [PAIR | {"original_text": "Short."}, {**PAIR, "language": "hi"},
             {"original_text": "x" * 61, "simplified_text": "y"}, {"original_text": "Pay Rs. 5.", "simplified_text": "Pay Rs. 6."}]
    d = c.post("/batch-predict", json={"items": items}).json()
    assert [r["status"] for r in d["results"]] == ["ok", "error", "error", "ok"]
    assert d["results"][1]["error"]["code"] == "unsupported_language" and d["results"][1]["error"]["supported_languages"] == ["en"]
    assert d["results"][2]["error"]["code"] == "input_too_long"
    assert (d["ok_count"], d["error_count"]) == (2, 2)


@pytest.mark.parametrize("body", [{"items": []}, {}, {"items": "x"}, {"items": [1]}, {"items": [{}]}, {"items": None},
                                  {"items": [PAIR], "extra": 1}, [PAIR]])
def test_batch_structure_errors(client, body):
    r = client.post("/batch-predict", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_batch_item_validation_error_names_the_item(client):
    r = client.post("/batch-predict", json={"items": [PAIR, {"original_text": " ", "simplified_text": "x"}]})
    assert r.status_code == 422
    assert any(d["field"] == "items.1.original_text" for d in r.json()["error"]["details"])


def test_batch_size_limit(make_client):
    c = make_client(max_batch=3)
    assert c.post("/batch-predict", json={"items": [PAIR] * 3}).status_code == 200
    r = c.post("/batch-predict", json={"items": [PAIR] * 4})
    assert r.status_code == 422 and r.json()["error"]["code"] == "batch_too_large"


# ------------------------------------------------------------------ /model-info
def test_model_info_without_model_or_results(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    m = ModelInfoResponse.model_validate(r.json())
    assert m.backbone == "FacebookAI/xlm-roberta-base" and m.model.available is False
    assert m.supported_languages == ["en"] and m.planned_languages_not_supported == ["hi"]
    assert m.score_range == [1.0, 10.0] and m.category_thresholds == {"preserved_min": 7.0, "partial_min": 4.0}
    assert m.evaluation.status == "no_saved_results" and m.evaluation.metrics is None
    assert m.baselines["tfidf"].available is False


def test_model_info_reports_only_real_saved_metrics(tmp_path, make_client):
    r = tmp_path / "results"
    r.mkdir()
    saved = {"fixture_note": "toy values written by a test", "pearson": 0.5}
    (r / "metrics.json").write_text(json.dumps(saved))
    e = make_client().get("/model-info").json()["evaluation"]
    assert e == {"status": "saved_results", "source": "metrics.json", "metrics": saved}


@pytest.mark.parametrize("content", ["{broken", "[1, 2]", '"text"'])
def test_model_info_unreadable_metrics(tmp_path, make_client, content):
    r = tmp_path / "results"
    r.mkdir()
    (r / "metrics.json").write_text(content)
    e = make_client().get("/model-info").json()["evaluation"]
    assert e["status"] == "unreadable" and e["metrics"] is None and str(tmp_path) not in json.dumps(e)


def test_custom_thresholds_are_reported(make_client):
    c = make_client(preserved_min=8.0, partial_min=5.0)
    assert c.get("/model-info").json()["category_thresholds"] == {"preserved_min": 8.0, "partial_min": 5.0}


def test_invalid_threshold_config_fails_at_startup(tmp_path):
    from app.main import create_app
    from conftest import make_settings
    with pytest.raises(ValueError):
        create_app(make_settings(tmp_path, preserved_min=4.0, partial_min=6.0))


# ------------------------------------------------------------------ errors, safety, schema
def test_unknown_route_and_wrong_method_use_the_error_shape(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/predict")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


def test_error_responses_never_leak_internals(tmp_path, make_client):
    c = make_client(max_chars=20, max_body_bytes=400)
    bodies = [{}, {"original_text": "x" * 21, "simplified_text": "y"}, {**PAIR, "language": "hi"}]
    texts = [c.post("/predict", json=b).text for b in bodies] + [c.get("/nope").text, c.post("/predict", content="{").text]
    for t in texts:
        assert "Traceback" not in t and "site-packages" not in t and str(tmp_path) not in t and "/home/" not in t


def test_openapi_documents_all_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert {"/health", "/predict", "/batch-predict", "/model-info"} <= set(r.json()["paths"])


def test_unhandled_exception_returns_generic_500(monkeypatch, tmp_path, make_client):
    import app.service as service
    c = make_client(raise_server_exceptions=False)
    monkeypatch.setattr(service.AnalysisService, "analyze", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret /etc/passwd")))
    r = c.post("/predict", json=PAIR)
    assert r.status_code == 500 and r.json() == {"error": {"code": "internal_error", "message": "Internal server error."}}


def test_explanation_failure_degrades_gracefully(monkeypatch, client):
    import app.service as service
    monkeypatch.setattr(service, "explain", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rules exploded")))
    r = client.post("/predict", json=PAIR)
    d = r.json()
    assert r.status_code == 200 and d["explanation_status"]["available"] is False
    assert d["legal_error_findings"] == [] and "rules exploded" not in r.text
