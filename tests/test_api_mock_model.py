"""API tests that use MockPredictor (a deterministic FAKE). They verify API plumbing, score validation and
category mapping. They say NOTHING about how a real trained model behaves."""
import json
import math

import pytest
from app.schemas import BatchResponse, PredictResponse
from conftest import PAIR, MockPredictor

pytestmark = pytest.mark.mock_model


@pytest.mark.parametrize("score,category", [
    (10.0, "Preserved"), (9.2, "Preserved"), (7.0, "Preserved"), (6.99, "Partially Preserved"), (5.0, "Partially Preserved"),
    (4.0, "Partially Preserved"), (3.99, "Not Preserved"), (1.0, "Not Preserved")])
def test_score_and_category_use_existing_thresholds(make_client, score, category):
    c = make_client(predictor=MockPredictor(scores=[score]))
    r = c.post("/predict", json=PAIR)
    assert r.status_code == 200
    d = PredictResponse.model_validate(r.json())
    assert d.lmp_status == "available" and d.predicted_lmp_score == score and d.preservation_category == category
    assert d.lmp_unavailable_reason is None and d.model_version == "mock-test-fixture"
    assert d.model.available is True and d.model.backbone == "mock-backbone"
    assert not any("No LMP score" in w for w in d.warnings)


def test_configured_thresholds_change_the_category(make_client):
    c = make_client(predictor=MockPredictor(scores=[7.5]), preserved_min=8.0, partial_min=6.0)
    assert c.post("/predict", json=PAIR).json()["preservation_category"] == "Partially Preserved"


def test_health_and_model_info_with_a_predictor(make_client):
    c = make_client(predictor=MockPredictor(scores=[5.0]))
    h = c.get("/health").json()
    assert h["ready_for_lmp_prediction"] is True and h["model"]["version"] == "mock-test-fixture"
    assert c.get("/model-info").json()["model"]["available"] is True


def test_rules_never_influence_the_score(make_client):
    # constant fake model: a harmless pair and a heavily changed pair must get the SAME score even though findings differ
    c = make_client(predictor=MockPredictor(fn=lambda o, s: 8.0))
    same = c.post("/predict", json={"original_text": "Rent is due.", "simplified_text": "Rent is due."}).json()
    bad = c.post("/predict", json={"original_text": "All tenants must pay Rs. 10,000 unless the landlord agrees.",
                                   "simplified_text": "Some tenants may pay Rs. 1,000."}).json()
    assert same["predicted_lmp_score"] == bad["predicted_lmp_score"] == 8.0
    assert same["legal_error_findings"] == [] and len(bad["legal_error_findings"]) >= 4


def test_predictor_receives_inputs_as_given(make_client):
    p = MockPredictor(scores=[5.0])
    c = make_client(predictor=p)
    c.post("/predict", json={**PAIR, "language": "EN"})
    assert p.calls == [([PAIR["original_text"]], [PAIR["simplified_text"]], ["en"])]


def test_unsupported_language_never_reaches_the_predictor(make_client):
    p = MockPredictor(scores=[5.0])
    c = make_client(predictor=p)
    assert c.post("/predict", json={**PAIR, "language": "hi"}).status_code == 422
    assert p.calls == []


def test_batch_uses_one_predictor_call_and_maps_scores_to_valid_items(make_client):
    p = MockPredictor(scores=[9.5, 2.0])
    c = make_client(predictor=p)
    items = [PAIR, {**PAIR, "language": "hi"}, {"original_text": "Pay Rs. 5.", "simplified_text": "Pay Rs. 6."}]
    r = c.post("/batch-predict", json={"items": items})
    b = BatchResponse.model_validate(r.json())
    assert len(p.calls) == 1 and len(p.calls[0][0]) == 2  # only the two supported items were scored, in one call
    assert [x.status for x in b.results] == ["ok", "error", "ok"]
    assert b.results[0].result.predicted_lmp_score == 9.5 and b.results[0].result.preservation_category == "Preserved"
    assert b.results[2].result.predicted_lmp_score == 2.0 and b.results[2].result.preservation_category == "Not Preserved"
    assert b.lmp_status == "available"


@pytest.mark.parametrize("raw", [[11.0], [0.5], [float("nan")], [math.inf], [-math.inf], [], [5.0, 5.0], ["high"], [None], 7])
def test_invalid_predictor_output_is_an_error_not_a_clipped_score(make_client, raw):
    c = make_client(predictor=MockPredictor(raw=raw), raise_server_exceptions=False)
    r = c.post("/predict", json=PAIR)
    assert r.status_code == 500 and r.json()["error"]["code"] == "inference_failed"
    assert "predicted_lmp_score" not in r.text


def test_predictor_exception_is_hidden_from_the_client(make_client):
    c = make_client(predictor=MockPredictor(raises=RuntimeError("CUDA OOM at /home/user/secret/model.bin")))
    for url, body in (("/predict", PAIR), ("/batch-predict", {"items": [PAIR]})):
        r = c.post(url, json=body)
        assert r.status_code == 500 and r.json() == {"error": {"code": "inference_failed", "message": "Model inference failed."}}
        assert "secret" not in r.text and "CUDA" not in r.text


def test_scores_are_floats_in_range_for_a_whole_batch(make_client):
    scores = [1.0, 3.3, 5.5, 7.7, 10.0]
    c = make_client(predictor=MockPredictor(scores=scores))
    d = c.post("/batch-predict", json={"items": [PAIR] * 5}).json()
    got = [r["result"]["predicted_lmp_score"] for r in d["results"]]
    assert got == scores and all(isinstance(g, float) and 1 <= g <= 10 for g in got)


def test_mock_is_not_mistaken_for_a_trained_model_in_default_config(client):
    assert client.get("/health").json()["ready_for_lmp_prediction"] is False  # default app has no predictor
