"""Dataset schema, validation and near-duplicate detection."""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

REQUIRED = ["pair_id", "source_document_id", "language", "legal_domain", "original_text",
            "simplified_text", "simplification_origin", "source_licence"]
OPTIONAL = ["source_url", "source_title", "annotator_id", "annotator_rating",
            "adjudicated_rating", "error_categories", "annotation_notes", "split"]
ORIGINS = {"human_written", "expert_reviewed", "synthetic"}
LANGUAGES = {"en", "hi"}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def validate(df: pd.DataFrame, require_ratings: bool = False) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errs: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for c in REQUIRED:
        bad = df[c].isna() | (df[c].astype(str).str.strip() == "")
        if bad.any():
            errs.append(f"empty '{c}' in rows {list(df.index[bad])[:5]}")
    if df["pair_id"].duplicated().any():
        errs.append("duplicate pair_id values")
    bad = ~df["language"].isin(LANGUAGES)
    if bad.any():
        errs.append(f"unsupported language in rows {list(df.index[bad])[:5]}")
    bad = ~df["simplification_origin"].isin(ORIGINS)
    if bad.any():
        errs.append(f"invalid simplification_origin in rows {list(df.index[bad])[:5]}")
    for col in ("annotator_rating", "adjudicated_rating"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = df[col].notna() & (v.isna() | (v < 1) | (v > 10))
            if bad.any():
                errs.append(f"{col} outside [1, 10] or non-numeric in rows {list(df.index[bad])[:5]}")
    if require_ratings:
        col = "adjudicated_rating" if "adjudicated_rating" in df.columns else "annotator_rating"
        if col not in df.columns or df[col].isna().any():
            errs.append(f"missing {col} values")
    return errs


def find_duplicates(df: pd.DataFrame, threshold: float = 0.9) -> list[tuple[str, str, float]]:
    """Exact and near-duplicate pairs (character-trigram Jaccard on original+simplified)."""
    def grams(s: str) -> set[str]:
        s = f"  {s} "
        return {s[i:i + 3] for i in range(len(s) - 2)}
    keys = (df["original_text"].map(normalize) + " || " + df["simplified_text"].map(normalize)).tolist()
    ids = df["pair_id"].tolist()
    gs = [grams(k) for k in keys]
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] == keys[j]:
                out.append((ids[i], ids[j], 1.0))
                continue
            inter = len(gs[i] & gs[j])
            sim = inter / max(1, len(gs[i] | gs[j]))
            if sim >= threshold:
                out.append((ids[i], ids[j], round(sim, 3)))
    return out


def drop_duplicates(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    drop = {b for _, b, _ in find_duplicates(df, threshold)}
    return df[~df["pair_id"].isin(drop)].reset_index(drop=True)


def unknown_error_categories(df: pd.DataFrame) -> dict:
    """Human `error_categories` labels outside the documented vocabulary (see app/explanations/types.py)."""
    from app.explanations.dataset import unknown_error_categories as _impl
    return _impl(df)

def validate_training_data(df: pd.DataFrame) -> list[str]:
    """Strict eligibility checks before training an LMP model."""
    errors = []

    required = [
        "pair_id", "source_document_id", "source_licence",
        "licence_status", "language", "original_text",
        "simplified_text", "simplification_origin",
        "adjudicated_rating", "adjudication_status",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        return [f"missing training columns: {missing}"]

    invalid_ids = df["source_document_id"].astype(str).str.strip().str.upper().isin(
        {"", "UNKNOWN", "UNRESOLVED", "NAN", "NONE"}
    )
    if invalid_ids.any():
        errors.append("unresolved source_document_id values")

    if not df["licence_status"].astype(str).str.strip().str.lower().eq("verified").all():
        errors.append("all source licences must have licence_status='verified'")

    invalid_licences = df["source_licence"].astype(str).str.strip().str.upper().isin(
        {"", "UNKNOWN", "UNVERIFIED", "NAN", "NONE"}
    )
    if invalid_licences.any():
        errors.append("missing or unverified source_licence values")

    if not df["simplification_origin"].isin({"human_written", "expert_reviewed"}).all():
        errors.append("synthetic or unsupported simplification origins are not eligible")

    ratings = pd.to_numeric(df["adjudicated_rating"], errors="coerce")
    if ratings.isna().any() or not ratings.between(1, 10).all():
        errors.append("every row needs an adjudicated_rating between 1 and 10")

    if not df["adjudication_status"].astype(str).str.strip().str.lower().eq("adjudicated").all():
        errors.append("every rating must have adjudication_status='adjudicated'")

    if df["pair_id"].duplicated().any():
        errors.append("duplicate pair_id values")

    for col in ("original_text", "simplified_text"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            errors.append(f"empty values in {col}")

    if not invalid_ids.any() and df["source_document_id"].nunique() < 3:
        errors.append("at least 3 distinct resolved source documents are required")

    return errors
