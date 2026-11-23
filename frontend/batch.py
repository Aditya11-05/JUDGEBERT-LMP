"""CSV batch analysis: validation, chunked submission to /batch-predict, result assembly and export.

Rules: no row is ever dropped silently (invalid rows stay in the results with their reason); every row keeps its
data-row number and optional input id; scores come from the backend only, and only when it reports 'available'.
"""
from __future__ import annotations
import io
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .api_client import MAX_BATCH_ITEMS, ApiClient, ApiClientError
from .presenters import score_view

REQUIRED = ("original_text", "simplified_text")
OPTIONAL = ("language", "context", "legal_domain")
ID_COLUMNS = ("pair_id", "id", "row_id")
MAX_UPLOAD_BYTES = 5_000_000
MAX_ROWS = int(os.environ.get("JUDGEBERT_UI_MAX_ROWS", "2000"))
MAX_CHARS = int(os.environ.get("JUDGEBERT_UI_MAX_CHARS", "5000"))  # mirrors the backend default per-field limit
DOMAIN_MAX = 100
BATCH_SIZE = MAX_BATCH_ITEMS
_LANG = re.compile(r"^[a-z]{2,3}([-_][a-z0-9]{2,8})?$")
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


@dataclass
class RowError:
    row_number: int
    input_id: str
    field: str
    message: str


@dataclass
class ParsedCsv:
    table: pd.DataFrame | None = None            # every data row, valid or not
    row_errors: list[RowError] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return 0 if self.table is None else int(self.table["valid"].sum())

    @property
    def n_invalid(self) -> int:
        return 0 if self.table is None else int((~self.table["valid"]).sum())


def parse_csv(data: bytes, supported_languages: list[str] | tuple[str, ...] = ("en",)) -> ParsedCsv:
    out = ParsedCsv()
    if len(data) > MAX_UPLOAD_BYTES:
        out.file_errors.append(f"The file is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
        return out
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        out.file_errors.append("The file is not valid UTF-8. Re-save it as 'CSV UTF-8'.")
        return out
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_filter=False)
    except pd.errors.EmptyDataError:
        out.file_errors.append("The file is empty.")
        return out
    except pd.errors.ParserError as e:
        out.file_errors.append(f"The file could not be parsed as CSV: {str(e)[:200]}")
        return out
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        out.file_errors.append(f"Missing required column(s): {', '.join(missing)}. "
                               f"Columns found: {', '.join(raw.columns) or '(none)'}.")
        return out
    if len(raw) == 0:
        out.file_errors.append("The file has a header but no data rows.")
        return out
    if len(raw) > MAX_ROWS:
        out.file_errors.append(f"The file has {len(raw)} rows; the limit is {MAX_ROWS}. Split it into smaller files.")
        return out

    id_col = next((c for c in ID_COLUMNS if c in raw.columns), None)
    supported = {s.lower() for s in supported_languages}
    tbl = pd.DataFrame({"row_number": range(1, len(raw) + 1)})
    tbl["input_id"] = raw[id_col].astype(str).str.strip() if id_col else ""
    for c in REQUIRED + OPTIONAL:
        tbl[c] = raw[c].astype(str) if c in raw.columns else ""
    langs, valid, why = [], [], []
    for _, r in tbl.iterrows():
        errs: list[tuple[str, str]] = []
        for c in REQUIRED:
            if not r[c].strip():
                errs.append((c, "is empty or whitespace-only"))
        for c in REQUIRED + ("context",):
            if len(r[c]) > MAX_CHARS:
                errs.append((c, f"is {len(r[c])} characters; the limit is {MAX_CHARS}"))
        if len(r["legal_domain"]) > DOMAIN_MAX:
            errs.append(("legal_domain", f"is longer than {DOMAIN_MAX} characters"))
        lang = r["language"].strip().lower() or "en"
        if not _LANG.match(lang):
            errs.append(("language", f"'{r['language'].strip()[:20]}' is not a language code such as 'en'"))
            code = lang
        else:
            code = re.split(r"[-_]", lang)[0]
            if code not in supported:
                errs.append(("language", f"'{code}' is not supported (supported: {', '.join(sorted(supported))})"))
        langs.append(code)
        valid.append(not errs)
        why.append("; ".join(f"{f} {m}" for f, m in errs))
        out.row_errors += [RowError(int(r["row_number"]), r["input_id"], f, m) for f, m in errs]
    tbl["language_used"], tbl["valid"], tbl["input_problems"] = langs, valid, why
    out.table = tbl
    return out


def build_items(table: pd.DataFrame) -> tuple[list[int], list[dict]]:
    """(row_numbers, API items) for the valid rows, in file order."""
    nums, items = [], []
    for _, r in table[table["valid"]].iterrows():
        it = {"original_text": r["original_text"], "simplified_text": r["simplified_text"], "language": r["language_used"]}
        if r["context"].strip():
            it["context"] = r["context"]
        if r["legal_domain"].strip():
            it["legal_domain"] = r["legal_domain"]
        nums.append(int(r["row_number"]))
        items.append(it)
    return nums, items


def chunked(seq: list, size: int = BATCH_SIZE):
    size = min(size, MAX_BATCH_ITEMS)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_batch(client: ApiClient, table: pd.DataFrame, include_baselines: bool = False,
              progress: Callable[[int, int], None] | None = None) -> dict[int, tuple[str, object]]:
    """Send valid rows in <=100-item requests. Returns {row_number: (status, payload)}.

    status: 'ok' (payload = predict response) | 'error' (payload = {'code','message'}) |
            'request_failed' / 'not_sent' (payload = message).
    """
    nums, items = build_items(table)
    out: dict[int, tuple[str, object]] = {}
    n_chunks = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
    aborted: str | None = None
    for ci, (cn, ci_items) in enumerate(zip(chunked(nums), chunked(items))):
        if aborted:
            for n in cn:
                out[n] = ("not_sent", aborted)
            continue
        try:
            resp = client.batch_predict(ci_items, include_baselines)
        except ApiClientError as e:
            for n in cn:
                out[n] = ("request_failed", e.message)
            if e.kind in ("offline", "timeout"):
                aborted = f"Not sent: {e.message}"
            if progress:
                progress(ci + 1, n_chunks)
            continue
        for n, r in zip(cn, resp["results"]):
            out[n] = ("ok", r["result"]) if r["status"] == "ok" else ("error", r["error"])
        if progress:
            progress(ci + 1, n_chunks)
    return out


def _findings_text(findings: list[dict]) -> str:
    parts = []
    for f in findings:
        o, s = f.get("original_span"), f.get("simplified_span")
        parts.append(f"[{f['evidence_strength']}] {f['error_type']} ({f['change_kind']}): "
                     f"'{o['text'] if o else ''}' -> '{s['text'] if s else ''}'")
    return " | ".join(parts)


def assemble(table: pd.DataFrame, outcomes: dict[int, tuple[str, object]]) -> pd.DataFrame:
    rows = []
    for _, r in table.iterrows():
        n = int(r["row_number"])
        row = {"row_number": n, "input_id": r["input_id"], "original_text": r["original_text"],
               "simplified_text": r["simplified_text"], "language": r["language_used"], "context": r["context"],
               "legal_domain": r["legal_domain"], "status": "", "error_code": "", "error_message": "",
               "lmp_status": "", "predicted_lmp_score": None, "preservation_category": "", "n_findings": None,
               "finding_types": "", "findings": "", "baseline_tfidf_score": None,
               "baseline_multilingual_embedding_score": None}
        if not r["valid"]:
            row.update(status="invalid_input", error_code="invalid_input", error_message=r["input_problems"])
        else:
            status, payload = outcomes.get(n, ("not_sent", "No result was recorded for this row."))
            if status == "ok":
                sv = score_view(payload)
                fs = payload["legal_error_findings"]
                row.update(status="ok", lmp_status=payload["lmp_status"], n_findings=len(fs),
                           predicted_lmp_score=float(payload["predicted_lmp_score"]) if sv.available else None,
                           preservation_category=sv.category or "",
                           finding_types=";".join(sorted({f["error_type"] for f in fs})), findings=_findings_text(fs))
                for name, b in payload["baseline_scores"]["scores"].items():
                    col = f"baseline_{name}_score"
                    if b["status"] == "available" and col in row:
                        row[col] = float(b["score"])
            elif status == "error":
                row.update(status="error", error_code=payload.get("code", ""), error_message=payload.get("message", ""))
            else:
                row.update(status=status, error_code=status, error_message=str(payload))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return {"rows": int(len(df)), "ok": int((df["status"] == "ok").sum()),
            "invalid_input": int((df["status"] == "invalid_input").sum()),
            "api_error": int((df["status"] == "error").sum()),
            "failed_or_not_sent": int(df["status"].isin(["request_failed", "not_sent"]).sum()),
            "with_lmp_score": int(df["predicted_lmp_score"].notna().sum()),
            "lmp_unavailable": int(((df["status"] == "ok") & (df["lmp_status"] == "unavailable")).sum())}


def _safe(v):
    return "'" + v if isinstance(v, str) and v.startswith(_FORMULA_START) else v


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 (with BOM for Excel). Text cells that start with = + - @ get a leading apostrophe so spreadsheets do not run them as formulas."""
    safe = df.copy()
    for c in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[c]):  # object OR the newer string dtype
            safe[c] = safe[c].map(_safe)
    return safe.to_csv(index=False).encode("utf-8-sig")
