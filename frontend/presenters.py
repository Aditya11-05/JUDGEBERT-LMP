"""Pure presentation helpers (no Streamlit, no network) so the display rules can be unit-tested."""
from __future__ import annotations
import html
from dataclasses import dataclass

UNAVAILABLE_BASE = "Legal Meaning Preservation score unavailable"
NOT_LEGAL_ADVICE = ("Research prototype. This tool does not provide legal advice and cannot establish legal conclusions. "
                    "Consult a qualified legal professional.")

ERROR_LABELS = {
    "obligation_permission_change": "Obligation / permission change", "negation_change": "Negation change",
    "date_or_deadline_change": "Date or deadline change", "bound_or_direction_change": "Bound or direction change",
    "monetary_amount_change": "Monetary amount change", "percentage_change": "Percentage change",
    "numerical_value_change": "Numerical value change", "frequency_change": "Frequency change",
    "party_or_entity_change": "Party or entity change", "scope_change": "Scope change",
    "condition_removed": "Condition removed", "condition_added": "Condition added",
    "condition_strength_change": "Condition strength change", "exception_removed": "Exception removed",
    "exception_added": "Exception added", "possible_omission_or_addition": "Possible omission or addition",
}
CHANGE_LABELS = {"changed": "Changed", "omitted": "Omitted in simplified text", "added": "Added in simplified text"}
STRENGTH_LABELS = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}
STRENGTH_HELP = ("Evidence strength is a categorical heuristic about how clear the textual evidence is. "
                 "It is not a probability and not a measure of legal importance.")
BASELINE_LABELS = {"tfidf": "TF-IDF cosine similarity", "multilingual_embedding": "Multilingual sentence embeddings"}


def error_label(error_type: str) -> str:
    return ERROR_LABELS.get(error_type, error_type.replace("_", " ").capitalize())


# ------------------------------------------------------------------ LMP score / status
@dataclass(frozen=True)
class ScoreView:
    available: bool
    headline: str
    score_text: str | None
    category: str | None
    detail: str


def unavailable_message(resp: dict) -> str:
    reason = resp.get("lmp_unavailable_reason")
    if reason in (None, "weights_missing"):
        return f"{UNAVAILABLE_BASE}: the trained transformer model has not yet been installed."
    msg = (resp.get("model") or {}).get("message") or "the model is not ready."
    return f"{UNAVAILABLE_BASE}: {msg[0].lower() + msg[1:] if msg else msg}"


def score_view(resp: dict) -> ScoreView:
    """The ONLY place a numeric score is turned into text; it requires status 'available' and a valid score."""
    s = resp.get("predicted_lmp_score")
    if resp.get("lmp_status") == "available" and isinstance(s, (int, float)) and not isinstance(s, bool) \
            and 1.0 <= s <= 10.0 and resp.get("preservation_category"):
        return ScoreView(True, "LMP score", f"{s:.1f} / 10", resp["preservation_category"],
                         "A high score is not a guarantee of legal correctness; review the findings below.")
    return ScoreView(False, "LMP score unavailable", None, None, unavailable_message(resp))


# ------------------------------------------------------------------ highlighting
def _valid_spans(text: str, spans) -> list[tuple[int, int]]:
    out = []
    for s in spans:
        if s and 0 <= s["start"] < s["end"] <= len(text) and text[s["start"]:s["end"]] == s["text"]:
            out.append((s["start"], s["end"]))
    out.sort()
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def highlight_html(text: str, spans, css_class: str = "chg") -> str:
    """Escape `text` and wrap the backend-provided spans in <mark>. Spans that do not match the text are ignored."""
    parts, pos = [], 0
    for a, b in _valid_spans(text, spans):
        parts += [html.escape(text[pos:a]), f'<mark class="{css_class}">', html.escape(text[a:b]), "</mark>"]
        pos = b
    parts.append(html.escape(text[pos:]))
    return "".join(parts)


def finding_rows(findings: list[dict]) -> list[dict]:
    rows = []
    for f in findings:
        o, s = f.get("original_span"), f.get("simplified_span")
        rows.append({"Category": error_label(f["error_type"]), "Change": CHANGE_LABELS.get(f["change_kind"], f["change_kind"]),
                     "Evidence": STRENGTH_LABELS[f["evidence_strength"]], "Original": o["text"] if o else "",
                     "Simplified": s["text"] if s else "", "Explanation": f["explanation"]})
    return rows


def finding_title(f: dict) -> str:
    o, s = f.get("original_span"), f.get("simplified_span")
    pair = f"'{o['text']}' → '{s['text']}'" if o and s else (f"'{o['text']}' (omitted)" if o else
                                                             (f"'{s['text']}' (added)" if s else "whole sentence"))
    return f"{STRENGTH_LABELS[f['evidence_strength']]} · {error_label(f['error_type'])} · {pair}"


# ------------------------------------------------------------------ baselines
def baseline_rows(resp: dict) -> list[dict]:
    rows = []
    for name, b in resp["baseline_scores"]["scores"].items():
        avail = b["status"] == "available"
        rows.append({"Baseline": BASELINE_LABELS.get(name, name), "Status": "Available" if avail else "Unavailable",
                     "Score (1-10)": f"{b['score']:.2f}" if avail else "—",
                     "Raw cosine similarity": f"{b['raw_similarity']:.3f}" if avail and isinstance(b.get("raw_similarity"), (int, float)) else "—",
                     "Note": "" if avail else (b.get("message") or b.get("reason") or "")})
    return rows


# ------------------------------------------------------------------ system status
def status_rows(health: dict, info: dict) -> list[dict]:
    """One row per component. 'Ready' requires the backend to say so; a model directory alone never counts."""
    m, ex = health["model"], health["explanations"]
    rows = [{"Component": "FastAPI backend", "State": "Connected", "Detail": "Responded to /health."},
            {"Component": "Trained transformer (XLM-R)",
             "State": "Ready" if health["ready_for_lmp_prediction"] and m["available"] else "Unavailable",
             "Detail": m.get("message") or (m.get("reason") or "")},
            {"Component": "Rule-based explanations", "State": "Available" if ex["available"] else "Unavailable",
             "Detail": "Languages: " + ", ".join(ex["supported_languages"])}]
    for name, b in health["baselines"].items():
        reason = b.get("reason")
        state = "Available" if b["available"] else ("Not implemented" if reason == "not_implemented" else "Unavailable")
        rows.append({"Component": "Baseline: " + BASELINE_LABELS.get(name, name), "State": state,
                     "Detail": "" if b["available"] else (b.get("message") or reason or "")})
    return rows
