"""Frontend API client tests.

Group 1 (MOCKED HTTP): httpx.MockTransport returns crafted payloads; they prove the client's handling of success, errors and
malformed responses, NOT that a real model works.
Group 2 (REAL BACKEND, in-process): requests are routed into the real FastAPI app (no model installed).
Group 3 (REAL BACKEND, real HTTP): a uvicorn subprocess on localhost.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from app.config import Settings
from app.main import create_app
from app.schemas import BatchResponse, HealthResponse, ModelInfoResponse, PredictResponse
from fastapi.testclient import TestClient
from frontend.api_client import DEFAULT_API_URL, ENV_URL, ApiClient, ApiClientError, api_url_from_env
import fixture_builders as fb

ROOT = Path(__file__).resolve().parents[1]
PAIR = ("The tenant must pay rent by the fifth day.", "The tenant may pay rent by the tenth day.")


def client_for(handler):
    return ApiClient("http://backend.test", transport=httpx.MockTransport(handler))


def json_ok(payload, status=200):
    # json.dumps (not httpx's json=) so non-standard tokens such as NaN can be sent, as a faulty backend might
    return lambda request: httpx.Response(status, content=json.dumps(payload), headers={"content-type": "application/json"})


# ------------------------------------------------------------------ fixtures stay faithful to the real backend schemas
def test_fixture_builders_match_backend_schemas():
    PredictResponse.model_validate(fb.predict_response())
    PredictResponse.model_validate(fb.predict_response(True, 8.4, "Preserved"))
    HealthResponse.model_validate(fb.health())
    HealthResponse.model_validate(fb.health(ready=True))
    ModelInfoResponse.model_validate(fb.model_info())
    BatchResponse.model_validate(fb.batch_response([1, 2, 3], errors={1: ("unsupported_language", "x")}))


# ------------------------------------------------------------------ Group 1: mocked HTTP
class TestMockedHttp:
    def test_predict_success_model_unavailable(self):
        r = client_for(json_ok(fb.predict_response())).predict(*PAIR)
        assert r["lmp_status"] == "unavailable" and r["predicted_lmp_score"] is None and r["preservation_category"] is None

    def test_predict_success_model_available_controlled_fixture(self):
        r = client_for(json_ok(fb.predict_response(True, 8.4, "Preserved"))).predict(*PAIR)
        assert r["lmp_status"] == "available" and r["predicted_lmp_score"] == 8.4 and r["preservation_category"] == "Preserved"

    def test_request_shape(self):
        seen = {}

        def handler(request):
            seen.update(method=request.method, path=request.url.path, body=json.loads(request.content))
            return httpx.Response(200, json=fb.predict_response())
        client_for(handler).predict("A.", "B.", "en", context="  ", legal_domain="", include_baselines=False)
        assert seen["method"] == "POST" and seen["path"] == "/predict"
        assert seen["body"] == {"original_text": "A.", "simplified_text": "B.", "language": "en", "include_baselines": False}
        client_for(handler).predict("A.", "B.", "en", context="clause 4", legal_domain="rental", include_baselines=True)
        assert seen["body"]["context"] == "clause 4" and seen["body"]["legal_domain"] == "rental" and seen["body"]["include_baselines"] is True

    def test_health_and_model_info(self):
        assert client_for(json_ok(fb.health())).health()["ready_for_lmp_prediction"] is False
        assert client_for(json_ok(fb.model_info())).model_info()["supported_languages"] == ["en"]

    # ---- offline / timeouts
    @pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"), httpx.NetworkError("down")])
    def test_backend_offline(self, exc):
        def handler(request):
            raise exc
        with pytest.raises(ApiClientError) as e:
            client_for(handler).predict(*PAIR)
        assert e.value.kind == "offline" and "backend.test" in e.value.message

    def test_read_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("slow")
        with pytest.raises(ApiClientError) as e:
            client_for(handler).predict(*PAIR)
        assert e.value.kind == "timeout"

    def test_real_connection_refused_is_reported_as_offline(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens here
        with pytest.raises(ApiClientError) as e:
            ApiClient(f"http://127.0.0.1:{port}", timeout=2.0).health()
        assert e.value.kind == "offline"

    # ---- HTTP errors
    def test_http_422_with_backend_error_body(self):
        body = {"error": {"code": "unsupported_language", "message": "Language 'hi' is not supported. Supported: en.",
                          "supported_languages": ["en"]}}
        with pytest.raises(ApiClientError) as e:
            client_for(json_ok(body, 422)).predict(*PAIR, language="hi")
        x = e.value
        assert (x.kind, x.status, x.code, x.supported_languages) == ("http", 422, "unsupported_language", ["en"])
        assert "not supported" in x.message

    def test_http_validation_error_details(self):
        body = {"error": {"code": "validation_error", "message": "The request is invalid.",
                          "details": [{"field": "original_text", "message": "must not be empty"}]}}
        with pytest.raises(ApiClientError) as e:
            client_for(json_ok(body, 422)).predict(*PAIR)
        assert e.value.details[0]["field"] == "original_text"

    @pytest.mark.parametrize("status,content", [(500, b"<html>Internal Server Error /home/x/app.py</html>"), (502, b""),
                                                (404, b"not found"), (503, b'{"detail": "x"}'), (500, b'{"error": "text"}')])
    def test_http_errors_without_backend_error_body(self, status, content):
        with pytest.raises(ApiClientError) as e:
            client_for(lambda r: httpx.Response(status, content=content)).health()
        assert e.value.kind == "http" and e.value.status == status
        assert e.value.message == f"The backend returned HTTP {status}."  # raw body is never shown

    # ---- invalid JSON / invalid schemas
    @pytest.mark.parametrize("content", [b"not json", b"", b"<html></html>", b"\xff\xfe"])
    def test_non_json_success_body(self, content):
        with pytest.raises(ApiClientError) as e:
            client_for(lambda r: httpx.Response(200, content=content)).predict(*PAIR)
        assert e.value.kind == "invalid_response"

    @pytest.mark.parametrize("payload", [[], "x", None, 7, {}, {"lmp_status": "available"}])
    def test_wrong_shape(self, payload):
        with pytest.raises(ApiClientError) as e:
            client_for(json_ok(payload)).predict(*PAIR)
        assert e.value.kind == "invalid_response"

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update(lmp_status="maybe"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=None, preservation_category=None),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=11.0, preservation_category="Preserved"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=0.5, preservation_category="Preserved"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=True, preservation_category="Preserved"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score="8", preservation_category="Preserved"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=float("nan"), preservation_category="Preserved"),
        lambda d: d.update(lmp_status="available", predicted_lmp_score=8.0, preservation_category="Great"),
        lambda d: d.update(predicted_lmp_score=7.5),                       # unavailable + score = fabricated score
        lambda d: d.update(preservation_category="Preserved"),            # unavailable + category
        lambda d: d["legal_error_findings"][0].update(evidence_strength="certain"),
        lambda d: d["legal_error_findings"][0].update(error_type=None),
        lambda d: d["legal_error_findings"][0].update(original_span={"text": "x"}),
        lambda d: d.update(legal_error_findings="none"),
        lambda d: d.pop("explanation_status"),
        lambda d: d.pop("baseline_scores"),
        lambda d: d["baseline_scores"].update(scores={"tfidf": {"status": "available", "score": None}}),
        lambda d: d["baseline_scores"].update(scores={"tfidf": {"status": "unavailable", "score": 5.0}}),
        lambda d: d["baseline_scores"].update(scores={"tfidf": {"status": "weird"}}),
        lambda d: d.update(warnings=[1, 2]),
        lambda d: d.pop("model"),
    ])
    def test_schema_violations_are_rejected(self, mutate):
        payload = fb.predict_response()
        mutate(payload)
        with pytest.raises(ApiClientError) as e:
            client_for(json_ok(payload)).predict(*PAIR)
        assert e.value.kind == "invalid_response"

    def test_a_score_is_never_accepted_next_to_unavailable_status(self):
        payload = fb.predict_response()
        payload["predicted_lmp_score"] = 9.1
        with pytest.raises(ApiClientError):
            client_for(json_ok(payload)).predict(*PAIR)

    def test_batch_response_checks(self):
        items = [{"original_text": "a", "simplified_text": "b"}] * 3
        ok = client_for(json_ok(fb.batch_response(items))).batch_predict(items)
        assert ok["count"] == 3
        bad_count = fb.batch_response(items[:2])
        with pytest.raises(ApiClientError) as e:
            client_for(json_ok(bad_count)).batch_predict(items)
        assert e.value.kind == "invalid_response"
        bad_index = fb.batch_response(items)
        bad_index["results"][1]["index"] = 5
        with pytest.raises(ApiClientError):
            client_for(json_ok(bad_index)).batch_predict(items)

    @pytest.mark.parametrize("n", [0, 101, 500])
    def test_batch_size_guard_on_the_client(self, n):
        with pytest.raises(ValueError):
            client_for(json_ok({})).batch_predict([{"original_text": "a", "simplified_text": "b"}] * n)

    def test_health_and_info_schema_violations(self):
        bad = fb.health()
        bad["status"] = "degraded"
        with pytest.raises(ApiClientError):
            client_for(json_ok(bad)).health()
        info = fb.model_info()
        info["evaluation"]["status"] = "???"
        with pytest.raises(ApiClientError):
            client_for(json_ok(info)).model_info()

    # ---- configuration
    def test_url_configuration(self, monkeypatch):
        monkeypatch.delenv(ENV_URL, raising=False)
        assert api_url_from_env() == DEFAULT_API_URL == "http://localhost:8000"
        monkeypatch.setenv(ENV_URL, "http://example.test:9000/")
        assert api_url_from_env() == "http://example.test:9000" and ApiClient().base_url == "http://example.test:9000"
        monkeypatch.setenv(ENV_URL, "   ")
        assert api_url_from_env() == DEFAULT_API_URL
        with pytest.raises(ValueError):
            ApiClient("ftp://nope")
        with pytest.raises(ValueError):
            ApiClient("localhost:8000")


# ------------------------------------------------------------------ Group 2: real FastAPI app, in-process transport
@pytest.fixture
def real_backend_client(tmp_path):
    settings = Settings(model_dir=tmp_path / "m", baseline_dir=tmp_path / "b", results_dir=tmp_path / "r")
    with TestClient(create_app(settings)) as tc:
        def handler(request: httpx.Request):
            r = tc.request(request.method, request.url.path, content=request.content,
                           headers={"content-type": "application/json"})
            return httpx.Response(r.status_code, content=r.content, headers={"content-type": "application/json"})
        yield ApiClient("http://in-process.test", transport=httpx.MockTransport(handler))


class TestAgainstRealBackendInProcess:
    def test_health_and_info(self, real_backend_client):
        h = real_backend_client.health()
        assert h["ready_for_lmp_prediction"] is False and h["model"]["reason"] == "weights_missing"
        i = real_backend_client.model_info()
        assert i["supported_languages"] == ["en"] and i["evaluation"]["status"] == "no_saved_results"

    def test_predict_model_unavailable_with_real_english_findings(self, real_backend_client):
        r = real_backend_client.predict(*PAIR)
        assert r["lmp_status"] == "unavailable" and r["predicted_lmp_score"] is None
        types = {f["error_type"] for f in r["legal_error_findings"]}
        assert {"obligation_permission_change", "date_or_deadline_change"} <= types

    def test_hindi_is_rejected_by_the_real_backend(self, real_backend_client):
        with pytest.raises(ApiClientError) as e:
            real_backend_client.predict("a", "b", language="hi")
        assert e.value.kind == "http" and e.value.code == "unsupported_language" and e.value.supported_languages == ["en"]

    def test_real_validation_error(self, real_backend_client):
        with pytest.raises(ApiClientError) as e:
            real_backend_client.predict("   ", "b")
        assert e.value.kind == "http" and e.value.status == 422 and e.value.details

    def test_real_batch(self, real_backend_client):
        items = [{"original_text": PAIR[0], "simplified_text": PAIR[1], "language": "en"},
                 {"original_text": "a", "simplified_text": "b", "language": "hi"}]
        d = real_backend_client.batch_predict(items)
        assert [r["status"] for r in d["results"]] == ["ok", "error"] and d["results"][1]["error"]["code"] == "unsupported_language"


# ------------------------------------------------------------------ Group 3: real HTTP to a real uvicorn process
@pytest.mark.live_server
class TestLiveServer:
    def test_client_over_real_http(self, live_url):
        c = ApiClient(live_url)
        assert c.health()["status"] == "ok"
        r = c.predict(*PAIR)
        assert r["lmp_status"] == "unavailable" and r["predicted_lmp_score"] is None
        assert any(f["error_type"] == "obligation_permission_change" for f in r["legal_error_findings"])
        with pytest.raises(ApiClientError) as e:
            c.predict("a", "b", language="hi")
        assert e.value.code == "unsupported_language"
