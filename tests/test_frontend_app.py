"""Streamlit UI tests using Streamlit's AppTest, which executes the REAL frontend/streamlit_app.py script.

Skipped when Streamlit is not installed in the interpreter running pytest (it is only installed in .venv here).
Group A (MOCKED backend): ApiClient methods are patched with crafted payloads, so these tests prove how the UI renders
each state; they do NOT show that a real trained model works.
Group B (REAL backend over HTTP): the app talks to a real uvicorn process on localhost (no model installed).
"""
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit", reason="Streamlit is not installed in this interpreter (see requirements-ui.txt)")
from streamlit.testing.v1 import AppTest  # noqa: E402

from frontend.api_client import ApiClient, ApiClientError  # noqa: E402
import fixture_builders as fb  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")
ORIG, SIMP = "The tenant must pay rent by the fifth day.", "The tenant may pay rent by the tenth day."
REQUIRED_MSG = "Legal Meaning Preservation score unavailable: the trained transformer model has not yet been installed."


def mocked(ready=False, predict=None, health=None, info=None):
    return [patch.object(ApiClient, "health", side_effect=health) if isinstance(health, Exception) else
            patch.object(ApiClient, "health", return_value=health or fb.health(ready)),
            patch.object(ApiClient, "model_info", return_value=info or fb.model_info(ready)),
            patch.object(ApiClient, "predict", side_effect=predict) if isinstance(predict, Exception) else
            patch.object(ApiClient, "predict", return_value=predict or fb.predict_response())]


def run_app(patches):
    for p in patches:
        p.start()
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
    except Exception:
        for p in patches:
            p.stop()
        raise
    return at


def stop(patches):
    for p in patches:
        p.stop()


def analyze(at, orig=ORIG, simp=SIMP):
    at.text_area(key="orig").set_value(orig)
    at.text_area(key="simp").set_value(simp)
    next(b for b in at.button if b.label == "Analyze").click()
    return at.run()


def md(at):
    return " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------ Group A: mocked backend
class TestMockedBackend:
    def test_app_starts_with_expected_structure(self):
        ps = mocked()
        try:
            at = run_app(ps)
            assert not at.exception
            assert [t.value for t in at.title] == ["JUDGEBERT-LMP"]
            assert [t.label for t in at.tabs] == ["Analyzer", "Batch analysis", "System status", "Research info", "Annotation Studio"]
            assert REQUIRED_MSG in md(at)  # banner shown as soon as the backend reports no model
        finally:
            stop(ps)

    def test_unavailable_model_shows_message_findings_and_no_score(self):
        ps = mocked(ready=False)
        try:
            at = analyze(run_app(ps))
            text = md(at)
            assert not at.exception
            assert REQUIRED_MSG in text and "Unavailable" in text
            assert "/ 10" not in text  # no score text anywhere
            labels = [e.label for e in at.expander]
            assert any("Obligation / permission change" in x and "'must' → 'may'" in x for x in labels)
            assert "Not legal advice" in text
            assert '<mark class="chg">' in text  # highlighted spans
        finally:
            stop(ps)

    def test_available_model_controlled_fixture_shows_score_and_category(self):
        ps = mocked(ready=True, predict=fb.predict_response(True, 8.4, "Preserved"))
        try:
            at = analyze(run_app(ps))
            text = md(at)
            assert not at.exception and "8.4 / 10" in text and "Preserved" in text
            assert REQUIRED_MSG not in text
            assert any("not a guarantee of legal correctness" in c.value for c in at.caption)
        finally:
            stop(ps)

    def test_payload_with_score_beside_unavailable_is_never_displayed(self):
        bad = fb.predict_response(False)
        bad["predicted_lmp_score"], bad["preservation_category"] = 9.9, "Preserved"
        # the real client would reject this; patching predict bypasses that, so this checks the UI layer on its own
        ps = mocked(predict=bad)
        try:
            at = analyze(run_app(ps))
            assert "9.9" not in md(at) and "/ 10" not in md(at)
        finally:
            stop(ps)

    def test_hindi_is_disabled(self):
        ps = mocked()
        try:
            at = run_app(ps)
            sb = at.selectbox(key="lang")
            assert sb.options == ["English", "Hindi (हिन्दी) — not yet supported"]
            sb.select("hi").run()
            assert any("not supported yet" in w.value for w in at.warning)
            assert next(b for b in at.button if b.label == "Analyze").disabled is True
            predict = ApiClient.predict
            assert predict.call_count == 0  # nothing was sent
        finally:
            stop(ps)

    def test_english_is_enabled_by_default(self):
        ps = mocked()
        try:
            at = run_app(ps)
            assert at.selectbox(key="lang").value == "en" and next(b for b in at.button if b.label == "Analyze").disabled is False
        finally:
            stop(ps)

    def test_empty_inputs_show_validation_errors_and_send_nothing(self):
        ps = mocked()
        try:
            at = analyze(run_app(ps), "", "   ")
            errs = [e.value for e in at.error]
            assert "Enter the original sentence." in errs and "Enter the simplified sentence." in errs
            assert ApiClient.predict.call_count == 0
        finally:
            stop(ps)

    def test_backend_offline_at_startup_and_on_submit(self):
        offline = ApiClientError("offline", "Could not reach the backend at http://localhost:8000. Is it running?")
        ps = mocked(health=offline, predict=offline)
        try:
            at = run_app(ps)
            assert not at.exception and any("not reachable" in e.value for e in at.error)
            at = analyze(at)
            assert any("Could not reach the backend" in e.value and "uvicorn app.main:app" in e.value for e in at.error)
            assert "/ 10" not in md(at)
        finally:
            stop(ps)

    def test_http_error_from_backend_is_shown_readably(self):
        err = ApiClientError("http", "The request is invalid.", 422, "validation_error",
                             [{"field": "original_text", "message": "must not be empty"}])
        ps = mocked(predict=err)
        try:
            at = analyze(run_app(ps))
            assert any("The request is invalid." in e.value and "original_text" in e.value for e in at.error)
        finally:
            stop(ps)

    def test_invalid_response_is_shown_as_an_error(self):
        err = ApiClientError("invalid_response", "The backend response was not in the expected format (x).")
        ps = mocked(predict=err)
        try:
            at = analyze(run_app(ps))
            assert any("not in the expected format" in e.value for e in at.error)
        finally:
            stop(ps)

    def test_status_tab_does_not_invent_metrics(self):
        ps = mocked()
        try:
            at = run_app(ps)
            assert any("No saved evaluation results exist yet" in i.value for i in at.info)
            assert any("does not mean a trained model is available" in s.value for s in at.success)
            assert not at.json  # no metrics block
        finally:
            stop(ps)

    def test_status_tab_shows_real_saved_metrics_when_the_backend_reports_them(self):
        info = fb.model_info(True, {"status": "saved_results", "source": "metrics.json", "metrics": {"pearson": 0.5, "note": "fixture"}})
        ps = mocked(ready=True, info=info)
        try:
            at = run_app(ps)
            assert len(at.json) >= 1 and "pearson" in at.json[0].value
        finally:
            stop(ps)

    def test_research_tab_distinguishes_implemented_from_planned(self):
        ps = mocked()
        try:
            at = run_app(ps)
            text = md(at)
            assert "JUDGEBERT-inspired" in text and "Similarity is not preservation" in text
            table = next(d.value for d in at.dataframe if "Component" in d.value.columns and "Status" in d.value.columns)
            status = dict(zip(table["Component"], table["Status"]))
            assert status["Hindi support"].startswith("Planned") and status["Trained XLM-RoBERTa LMP regressor"].startswith("Planned")
            assert status["Rule-based English explanations"].startswith("Implemented")
        finally:
            stop(ps)


# ------------------------------------------------------------------ Group B: real backend over HTTP
@pytest.mark.live_server
class TestRealBackend:
    def test_analyzer_against_live_backend(self, live_url, monkeypatch):
        monkeypatch.setenv("JUDGEBERT_API_URL", live_url)
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception and REQUIRED_MSG in md(at)
        at = analyze(at)
        text = md(at)
        assert not at.exception and "/ 10" not in text and REQUIRED_MSG in text
        labels = [e.label for e in at.expander]
        assert any("Obligation / permission change" in x for x in labels) and any("Date or deadline change" in x for x in labels)

    def test_offline_when_url_points_nowhere(self, monkeypatch):
        monkeypatch.setenv("JUDGEBERT_API_URL", "http://127.0.0.1:9")
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception and any("not reachable" in e.value for e in at.error)
