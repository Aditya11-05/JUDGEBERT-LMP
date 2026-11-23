"""JUDGEBERT-LMP research demo. Run from the repository root:  streamlit run frontend/streamlit_app.py

Everything shown comes from the FastAPI backend over HTTP (JUDGEBERT_API_URL, default http://localhost:8000).
Nothing here computes, estimates or defaults an LMP score.
"""
from __future__ import annotations
import hashlib
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `frontend` imports work with `streamlit run`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frontend import batch  # noqa: E402
from frontend import annotation  # noqa: E402
from frontend import presenters as P  # noqa: E402
from frontend.api_client import MAX_BATCH_ITEMS, ApiClient, ApiClientError, api_url_from_env  # noqa: E402

LANG_NAMES = {"en": "English", "hi": "Hindi (हिन्दी)"}
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1180px;}
h1 {margin-bottom: 0;} .subtitle {color:#4a5a6a; font-size:1.05rem; margin:0 0 .8rem 0;}
.banner {background:#fff7e0; border-left:5px solid #b7791f; padding:.55rem .9rem; border-radius:4px; color:#5c3d00; margin:.4rem 0 1rem 0;}
.card {background:#f3f6fa; border:1px solid #d5dde6; border-radius:8px; padding:.9rem 1.1rem; min-height:6.2rem;}
.card .label {font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:#4a5a6a;}
.card .value {font-size:1.9rem; font-weight:650; color:#14324f; line-height:1.25;}
.card .value.na {font-size:1.25rem; color:#7a3b00;}
.card .sub {font-size:.88rem; color:#38485a;}
.unavail {background:#fdf0e3; border:1px solid #e0b27a; border-left:5px solid #c25e00; border-radius:6px; padding:.7rem 1rem; color:#5a2c00; margin:.6rem 0;}
.textbox {background:#fff; border:1px solid #d5dde6; border-radius:6px; padding:.7rem .9rem; white-space:pre-wrap; line-height:1.55; min-height:4rem;}
mark.chg {background:#ffe08a; color:#1b2733; padding:0 .15em; border-radius:3px; border-bottom:2px solid #b7791f;}
.pill {display:inline-block; padding:.05rem .55rem; border-radius:999px; font-size:.82rem; font-weight:600; border:1px solid transparent;}
.pill.ok {background:#e3f4e8; color:#14532d; border-color:#9bd3ad;} .pill.bad {background:#fde8e8; color:#8a1c1c; border-color:#ee9b9b;}
.pill.warn {background:#fff1d6; color:#7a4b00; border-color:#e8c27a;} .pill.info {background:#e6eef7; color:#1f3d63; border-color:#a9c2de;}
</style>
"""


def pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def card(label: str, value: str, sub: str = "", na: bool = False) -> str:
    return (f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value{" na" if na else ""}">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>')


# ------------------------------------------------------------------ backend access
def get_client() -> ApiClient | None:
    try:
        return ApiClient()
    except ValueError as e:
        st.error(f"Configuration problem: {e}")
        return None


def describe(e: ApiClientError) -> str:
    if e.kind == "offline":
        return f"{e.message} Start it with `uvicorn app.main:app --port 8000` or set JUDGEBERT_API_URL."
    if e.kind == "http" and e.details:
        return e.message + " " + "; ".join(f"{d.get('field', '?')}: {d.get('message', '')}" for d in e.details[:5])
    return e.message


def refresh_status() -> None:
    client = get_client()
    state = {"health": None, "info": None, "error": None, "info_error": None}
    if client is None:
        state["error"] = "Invalid backend URL."
    else:
        try:
            state["health"] = client.health()
        except ApiClientError as e:
            state["error"] = describe(e)
        if state["health"] is not None:
            try:
                state["info"] = client.model_info()
            except ApiClientError as e:
                state["info_error"] = describe(e)
    st.session_state["status"] = state


def languages() -> tuple[list[str], list[str]]:
    info = (st.session_state.get("status") or {}).get("info")
    if info:
        return list(info["supported_languages"]), list(info["planned_languages_not_supported"])
    return ["en"], ["hi"]  # conservative fallback until the backend has answered


# ------------------------------------------------------------------ rendering
def render_result(last: dict) -> None:
    if last.get("error"):
        st.error(last["error"])
        return
    r, orig, simp = last["resp"], last["orig"], last["simp"]
    sv = P.score_view(r)
    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)
    if sv.available:
        c1.markdown(card("LMP score", sv.score_text, "Predicted by the trained model"), unsafe_allow_html=True)
        c2.markdown(card("Preservation category", sv.category, "Thresholds are proposed defaults"), unsafe_allow_html=True)
    else:
        c1.markdown(card("LMP score", "Unavailable", "No score was produced", na=True), unsafe_allow_html=True)
        c2.markdown(card("Preservation category", "Unavailable", "Needs a valid LMP score", na=True), unsafe_allow_html=True)
    n = len(r["legal_error_findings"])
    c3.markdown(card("Rule-based findings", str(n), "Heuristic hints, independent of the LMP model"), unsafe_allow_html=True)
    if sv.available:
        st.caption(sv.detail)
    else:
        st.markdown(f'<div class="unavail"><b>{html.escape(sv.detail)}</b><br>The rule-based findings below are real output '
                    "from the backend; they are not a substitute for an LMP score and no score is estimated from them.</div>",
                    unsafe_allow_html=True)
    st.markdown(f'<div class="banner"><b>Not legal advice.</b> {html.escape(P.NOT_LEGAL_ADVICE)}</div>', unsafe_allow_html=True)

    st.markdown("#### Original and simplified text")
    fs = r["legal_error_findings"]
    left, right = st.columns(2)
    left.markdown("**Original**")
    left.markdown(f'<div class="textbox">{P.highlight_html(orig, [f.get("original_span") for f in fs])}</div>', unsafe_allow_html=True)
    right.markdown("**Simplified**")
    right.markdown(f'<div class="textbox">{P.highlight_html(simp, [f.get("simplified_span") for f in fs])}</div>', unsafe_allow_html=True)
    st.caption("Highlighted spans are phrases the rule-based module flagged as potentially meaning-changing.")

    st.markdown("#### Legal-error findings")
    if not r["explanation_status"]["available"]:
        st.warning(r["explanation_status"].get("message") or "Explanations are unavailable for this request.")
    elif not fs:
        st.info(r.get("explanation_summary") or "No potential legal-meaning change was detected by the rule set.")
    else:
        st.caption(P.STRENGTH_HELP)
        for f in fs:
            with st.expander(P.finding_title(f)):
                st.markdown(f"**Category:** {P.error_label(f['error_type'])}  \n**Change type:** {P.CHANGE_LABELS.get(f['change_kind'], f['change_kind'])}  \n"
                            f"**Evidence strength:** {P.STRENGTH_LABELS[f['evidence_strength']]}  \n**Rule:** `{f.get('rule_id', '')}`")
                st.write(f["explanation"])
        with st.expander("All findings as a table"):
            st.dataframe(pd.DataFrame(P.finding_rows(fs)), hide_index=True)

    render_baselines(r)
    for w in r["warnings"]:
        st.warning(w)
    with st.expander("Limitations of this analysis"):
        for x in r["limitations"]:
            st.markdown(f"- {x}")


def render_baselines(r: dict) -> None:
    st.markdown("#### Baseline comparison")
    b = r["baseline_scores"]
    if not b["requested"]:
        st.caption("Baselines were not requested. Tick 'Also request baseline scores' before analysing.")
        return
    st.dataframe(pd.DataFrame(P.baseline_rows(r)), hide_index=True)
    if any(x["status"] == "available" for x in b["scores"].values()):
        st.caption("Baselines are generic similarity measures mapped onto the 1-10 scale using a train-only fit. They measure "
                   "textual similarity, not legal meaning preservation, and are not predictions of the LMP model.")
    else:
        st.info("No fitted baseline is available on the backend, so no baseline scores can be shown.")


# ------------------------------------------------------------------ tabs
def tab_analyzer() -> None:
    supported, planned = languages()
    options = supported + [x for x in planned if x not in supported]
    lang = st.selectbox("Language", options, key="lang",
                        format_func=lambda c: LANG_NAMES.get(c, c) + ("" if c in supported else " — not yet supported"))
    unsupported = lang not in supported
    if unsupported:
        st.warning(f"{LANG_NAMES.get(lang, lang)} is not supported yet (no rules or tests exist for it). Choose English to analyse text.")
    with st.form("analyze_form"):
        a, b = st.columns(2)
        orig = a.text_area("Original legal sentence", height=170, max_chars=batch.MAX_CHARS, key="orig",
                           placeholder="The tenant must pay rent by the fifth day of each month.")
        simp = b.text_area("Simplified or reformulated sentence", height=170, max_chars=batch.MAX_CHARS, key="simp",
                           placeholder="The tenant may pay rent by the tenth day of each month.")
        with st.expander("Optional context and legal domain"):
            ctx = st.text_area("Surrounding context", height=90, max_chars=batch.MAX_CHARS, key="ctx")
            dom = st.text_input("Legal domain", max_chars=batch.DOMAIN_MAX, key="dom", placeholder="e.g. rental agreement")
            st.caption("Both fields are sent to the backend but are not used by the current pipeline; the response says so.")
        inc = st.checkbox("Also request baseline scores (shown only if fitted baselines exist)", key="inc_base")
        go = st.form_submit_button("Analyze", type="primary", disabled=unsupported)
    if go:
        problems = []
        if not orig.strip():
            problems.append("Enter the original sentence.")
        if not simp.strip():
            problems.append("Enter the simplified sentence.")
        if problems:
            for p in problems:
                st.error(p)
        else:
            client = get_client()
            if client is not None:
                with st.spinner("Analysing…"):
                    try:
                        resp = client.predict(orig, simp, lang, ctx, dom, inc)
                        st.session_state["last"] = {"resp": resp, "orig": orig, "simp": simp}
                    except ApiClientError as e:
                        st.session_state["last"] = {"error": describe(e)}
    if st.session_state.get("last"):
        render_result(st.session_state["last"])


def tab_batch() -> None:
    st.markdown("Upload a CSV with the columns `original_text` and `simplified_text`. Optional columns: `language`, `context`, "
                "`legal_domain`, and an id column (`pair_id`, `id` or `row_id`) that is carried through to the results.")
    st.caption(f"Limits: {batch.MAX_UPLOAD_BYTES // 1_000_000} MB, {batch.MAX_ROWS} rows, {batch.MAX_CHARS} characters per text field. "
               f"Requests are sent in groups of at most {MAX_BATCH_ITEMS}.")
    up = st.file_uploader("CSV file", type=["csv"], key="csv_upload")
    if up is None:
        return
    data = up.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    supported, _ = languages()
    parsed = batch.parse_csv(data, supported)
    if parsed.file_errors:
        for e in parsed.file_errors:
            st.error(e)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in file", len(parsed.table))
    c2.metric("Valid rows", parsed.n_valid)
    c3.metric("Invalid rows", parsed.n_invalid)
    if parsed.row_errors:
        st.warning("Invalid rows are not sent to the backend, but they are kept in the results with their reason.")
        st.dataframe(pd.DataFrame([vars(e) for e in parsed.row_errors]).rename(columns={
            "row_number": "Row", "input_id": "Id", "field": "Field", "message": "Problem"}), hide_index=True)
    inc = st.checkbox("Also request baseline scores (if available)", key="batch_inc")
    if st.button("Run batch analysis", type="primary", disabled=parsed.n_valid == 0):
        client = get_client()
        if client is not None:
            bar = st.progress(0.0, text="Sending…")
            out = batch.run_batch(client, parsed.table, inc, lambda d, t: bar.progress(d / max(t, 1), text=f"Batch {d} of {t}"))
            bar.empty()
            st.session_state["batch_result"] = {"digest": digest, "df": batch.assemble(parsed.table, out)}
    res = st.session_state.get("batch_result")
    if res and res["digest"] == digest:
        df, s = res["df"], batch.summarize(res["df"])
        st.markdown("#### Results")
        m = st.columns(4)
        m[0].metric("Analysed", s["ok"])
        m[1].metric("Invalid input", s["invalid_input"])
        m[2].metric("Backend errors / not sent", s["api_error"] + s["failed_or_not_sent"])
        m[3].metric("With LMP score", s["with_lmp_score"])
        if s["ok"] and s["lmp_unavailable"] == s["ok"]:
            st.markdown(f'<div class="unavail"><b>{html.escape(P.UNAVAILABLE_BASE)}.</b> The backend has no usable model, so the score '
                        "columns are empty. Rule-based findings are still included.</div>", unsafe_allow_html=True)
        st.dataframe(df, hide_index=True)
        st.download_button("Download results (CSV)", batch.to_csv_bytes(df), file_name="judgebert_lmp_results.csv", mime="text/csv")


def tab_status() -> None:
    if st.button("Refresh status", key="refresh_status"):
        refresh_status()
    s = st.session_state.get("status")
    if s is None:
        refresh_status()
        s = st.session_state["status"]
    if s["health"] is None:
        st.error(s["error"] or "The backend could not be reached.")
        st.caption(f"Backend URL: {api_url_from_env()}")
        return
    h, info = s["health"], s["info"]
    st.success("API connected — this only means the web service is up. It does not mean a trained model is available.")
    if info is None:
        st.warning(s["info_error"] or "Model details are unavailable.")
        return
    st.dataframe(pd.DataFrame(P.status_rows(h, info)), hide_index=True)
    st.markdown(f"**Backbone configured:** `{info['backbone']}`  \n**Supported languages:** {', '.join(info['supported_languages'])}  \n"
                f"**Planned, not supported:** {', '.join(info['planned_languages_not_supported']) or 'none'}  \n"
                f"**Model version:** {info['model'].get('version') or 'none installed'}")
    th = info.get("category_thresholds") or {}
    if th:
        st.caption(f"Category thresholds (proposed defaults): Preserved ≥ {th.get('preserved_min')}, Partially Preserved ≥ {th.get('partial_min')}.")
    ev = info["evaluation"]
    st.markdown("#### Evaluation results")
    if ev["status"] == "saved_results":
        st.caption(f"Loaded from the backend's saved results file ({ev.get('source')}).")
        st.json(ev["metrics"])
    elif ev["status"] == "unreadable":
        st.warning("A saved results file exists but could not be read.")
    else:
        st.info("No saved evaluation results exist yet, so no metrics are shown. No model has been trained or evaluated.")
    st.markdown("#### Baselines")
    for name, b in info["baselines"].items():
        st.markdown(f"- **{P.BASELINE_LABELS.get(name, name)}:** " + ("available" if b["available"] else f"unavailable — {b.get('message') or b.get('reason')}"))
        if b.get("metadata"):
            st.json(b["metadata"], expanded=False)


def tab_research() -> None:
    st.markdown("""
**Legal Meaning Preservation (LMP).** Legal text simplification should make wording easier without changing what a clause
legally means. LMP asks: does the simplified sentence keep the original's duties, rights, conditions, exceptions and scope?
The task was introduced by Beauchemin et al. (EMNLP 2025, JUDGEBERT), who built a French insurance dataset and a transformer
that predicts human LMP ratings. This project is a **JUDGEBERT-inspired** extension, not the official model.

**Similarity is not preservation.** "The tenant **must** pay by the fifth" and "The tenant **may** pay by the tenth" are almost
identical as text, so generic similarity scores rate them highly, yet an obligation became a permission and the deadline moved.
The research question is whether a model trained on human LMP ratings, in English and one Indian language, tracks such changes
better than generic similarity baselines.

**The rule-based explanation module** looks for specific change types (modals, negation, dates, amounts, scope, conditions,
exceptions, parties). It is transparent but lexical: it cannot understand meaning, it misses antonyms and paraphrased conditions,
and it is **not** an explanation of what the transformer computes. Its findings never feed into the LMP score.
""")
    st.markdown("#### What exists today")
    st.dataframe(pd.DataFrame([
        {"Component": "Rule-based English explanations", "Status": "Implemented, unit-tested; accuracy on human labels not measured"},
        {"Component": "FastAPI backend and this interface", "Status": "Implemented"},
        {"Component": "TF-IDF baseline", "Status": "Code implemented; no fitted artifact unless you create one from real training data"},
        {"Component": "Trained XLM-RoBERTa LMP regressor", "Status": "Planned — not trained, no inference code"},
        {"Component": "Multilingual-embedding baseline", "Status": "Planned — not implemented"},
        {"Component": "Hindi support", "Status": "Planned — not supported"},
        {"Component": "Human-rated dataset (≥300 pairs)", "Status": "Planned — does not exist yet"},
        {"Component": "Experiments and evaluation results", "Status": "Planned — none exist"},
    ]), hide_index=True)
    st.markdown("#### Current limitations")
    st.markdown("- No trained model, so **no LMP scores can be produced**.\n- No human-annotated data; no evaluation numbers exist.\n"
                "- English only; Hindi is rejected.\n- Rule-based findings are heuristics with known false positives and false negatives.\n"
                "- Sentence-level only: `context` is accepted but not used.\n- Research prototype; not legal advice.")


def main() -> None:
    st.set_page_config(page_title="JUDGEBERT-LMP", page_icon="⚖️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("JUDGEBERT-LMP")
    st.markdown('<p class="subtitle">Explainable legal meaning preservation — a research prototype (CSET 346)</p>', unsafe_allow_html=True)
    if "status" not in st.session_state:
        refresh_status()
    s = st.session_state["status"]
    with st.sidebar:
        st.markdown("### Backend")
        st.caption(api_url_from_env())
        if s["health"] is None:
            st.markdown(pill("Offline", "bad"), unsafe_allow_html=True)
        else:
            st.markdown(pill("API connected", "ok"), unsafe_allow_html=True)
            ready = s["health"]["ready_for_lmp_prediction"]
            st.markdown(pill("Model ready" if ready else "No trained model", "ok" if ready else "warn"), unsafe_allow_html=True)
            st.markdown(pill("Explanations on" if s["health"]["explanations"]["available"] else "Explanations off",
                             "ok" if s["health"]["explanations"]["available"] else "bad"), unsafe_allow_html=True)
        if st.button("Check connection", key="side_refresh"):
            refresh_status()
            st.rerun()
        st.caption("Set JUDGEBERT_API_URL to point at another backend.")
    if s["health"] is None:
        st.error(f"The backend is not reachable. {s['error'] or ''}")
    elif not s["health"]["ready_for_lmp_prediction"]:
        st.markdown(f'<div class="unavail"><b>{html.escape(P.UNAVAILABLE_BASE)}: the trained transformer model has not yet been installed.</b>'
                    " You can still analyse English text and see the rule-based findings.</div>", unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.tabs(
        ["Analyzer", "Batch analysis", "System status", "Research info", "Annotation Studio"]
    )
    with t1:
        tab_analyzer()
    with t2:
        tab_batch()
    with t3:
        tab_status()
    with t4:
        tab_research()
    with t5:
        annotation.render_annotation_studio()


main()
