"""Pure display-logic tests (no network, no Streamlit)."""
import pytest
from frontend import presenters as P
import fixture_builders as fb


def test_unavailable_state_uses_the_required_message_and_no_number():
    v = P.score_view(fb.predict_response(False))
    assert v.available is False and v.score_text is None and v.category is None
    assert v.detail == "Legal Meaning Preservation score unavailable: the trained transformer model has not yet been installed."


def test_available_state_shows_score_and_category():
    v = P.score_view(fb.predict_response(True, 8.44, "Preserved"))
    assert v.available and v.score_text == "8.4 / 10" and v.category == "Preserved" and "not a guarantee" in v.detail


@pytest.mark.parametrize("patch", [
    {"lmp_status": "unavailable", "predicted_lmp_score": 9.0, "preservation_category": "Preserved"},  # score beside 'unavailable'
    {"lmp_status": "available", "predicted_lmp_score": None},
    {"lmp_status": "available", "predicted_lmp_score": 11.0},
    {"lmp_status": "available", "predicted_lmp_score": 0.0},
    {"lmp_status": "available", "predicted_lmp_score": True},
    {"lmp_status": "available", "predicted_lmp_score": 5.0, "preservation_category": None},
    {},
])
def test_no_score_is_rendered_unless_the_payload_is_a_valid_available_prediction(patch):
    r = fb.predict_response(False)
    r.update(patch)
    if patch.get("lmp_status") == "available" and patch.get("predicted_lmp_score") == 5.0:
        r["preservation_category"] = None
    v = P.score_view(r)
    assert v.available is False and v.score_text is None


def test_other_unavailable_reasons_are_explained():
    r = fb.predict_response(False, reason="loader_not_implemented")
    r["model"]["message"] = "Model artifacts were found, but transformer inference is not integrated yet."
    assert P.score_view(r).detail == ("Legal Meaning Preservation score unavailable: model artifacts were found, "
                                      "but transformer inference is not integrated yet.")


def test_highlight_escapes_html_and_marks_spans():
    text = "Pay <b>rent</b> & must go"
    i = text.index("must")
    spans = [{"text": "must", "start": i, "end": i + 4}]
    out = P.highlight_html(text, spans)
    assert out == 'Pay &lt;b&gt;rent&lt;/b&gt; &amp; <mark class="chg">must</mark> go' and "<b>" not in out


def test_highlight_ignores_spans_that_do_not_match_the_text_and_merges_overlaps():
    t = "abcdefgh"
    assert P.highlight_html(t, [{"text": "zzz", "start": 1, "end": 4}, None, {"text": "x", "start": 5, "end": 99}]) == t
    merged = P.highlight_html(t, [{"text": "bcd", "start": 1, "end": 4}, {"text": "def", "start": 3, "end": 6}])
    assert merged == 'a<mark class="chg">bcdef</mark>gh' or merged.count("<mark") == 2  # overlap handling never loses text
    assert "".join(c for c in merged if True).replace('<mark class="chg">', "").replace("</mark>", "") == t


def test_finding_rows_and_titles():
    f = fb.finding()
    row = P.finding_rows([f])[0]
    assert row["Category"] == "Obligation / permission change" and row["Evidence"] == "Strong" and row["Original"] == "must"
    assert P.finding_title(f) == "Strong · Obligation / permission change · 'must' → 'may'"
    assert "(omitted)" in P.finding_title(fb.finding(s=None, kind="omitted"))
    assert "(added)" in P.finding_title(fb.finding(o=None, kind="added"))
    assert "whole sentence" in P.finding_title(fb.finding(o=None, s=None))
    assert P.error_label("some_new_type") == "Some new type"


def test_baseline_rows_are_honest_about_missing_baselines():
    r = fb.predict_response(False, requested=True, baselines={
        "tfidf": {"status": "unavailable", "score": None, "reason": "artifact_missing", "message": "No fitted TF-IDF baseline artifact is installed."},
        "multilingual_embedding": {"status": "unavailable", "score": None, "reason": "not_implemented"}})
    rows = P.baseline_rows(r)
    assert [x["Status"] for x in rows] == ["Unavailable", "Unavailable"] and all(x["Score (1-10)"] == "—" for x in rows)
    r["baseline_scores"]["scores"]["tfidf"] = {"status": "available", "score": 6.25, "raw_similarity": 0.8123}
    ok = P.baseline_rows(r)[0]
    assert ok["Score (1-10)"] == "6.25" and ok["Raw cosine similarity"] == "0.812" and ok["Baseline"] == "TF-IDF cosine similarity"


def test_status_rows_never_claim_a_model_is_ready_without_the_backend_saying_so():
    rows = {r["Component"]: r for r in P.status_rows(fb.health(ready=False), fb.model_info())}
    assert rows["FastAPI backend"]["State"] == "Connected" and rows["Trained transformer (XLM-R)"]["State"] == "Unavailable"
    assert rows["Baseline: Multilingual sentence embeddings"]["State"] == "Not implemented"
    assert rows["Baseline: TF-IDF cosine similarity"]["State"] == "Unavailable"
    ready = {r["Component"]: r for r in P.status_rows(fb.health(ready=True), fb.model_info(True))}
    assert ready["Trained transformer (XLM-R)"]["State"] == "Ready"
    inconsistent = fb.health(ready=False)
    inconsistent["model"]["available"] = True  # model dir 'exists' but backend says not ready
    assert {r["Component"]: r for r in P.status_rows(inconsistent, fb.model_info())}["Trained transformer (XLM-R)"]["State"] == "Unavailable"
