"""Integration with the dataset schema.

Adds rule-derived columns next to the human `error_categories` column and scores the rules against
those human labels. This measures the EXPLANATION MODULE only; it is separate from LMP regression accuracy.
"""
from __future__ import annotations
import json
import pandas as pd
from .engine import SUPPORTED_LANGUAGES, find_changes
from .types import ERROR_TYPES, HUMAN_ONLY_TYPES

VOCAB = set(ERROR_TYPES) | set(HUMAN_ONLY_TYPES)


def parse_categories(value) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {c.strip() for c in str(value).split(";") if c.strip() and c.strip() != "none"}


def unknown_error_categories(df: pd.DataFrame, col: str = "error_categories") -> dict[str, list]:
    """Rows whose human labels fall outside the documented vocabulary (returns {label: [row indices]})."""
    bad: dict[str, list] = {}
    if col in df.columns:
        for i, v in df[col].items():
            for c in parse_categories(v) - VOCAB:
                bad.setdefault(c, []).append(i)
    return bad


def annotate_dataframe(df: pd.DataFrame, min_strength: str = "weak") -> pd.DataFrame:
    """Add rule_supported, rule_error_types (';'-joined, same format as error_categories), rule_n_findings, rule_findings (JSON)."""
    out = df.copy()
    sup, types, counts, blobs = [], [], [], []
    for _, r in out.iterrows():
        if r["language"] not in SUPPORTED_LANGUAGES:
            sup.append(False), types.append(None), counts.append(None), blobs.append(None)
            continue
        fs = find_changes(r["original_text"], r["simplified_text"], r["language"], min_strength)
        sup.append(True)
        types.append(";".join(sorted({f.error_type for f in fs})))
        counts.append(len(fs))
        blobs.append(json.dumps([f.to_dict() for f in fs], ensure_ascii=False))
    out["rule_supported"], out["rule_error_types"] = sup, types
    out["rule_n_findings"], out["rule_findings"] = counts, blobs
    return out


def category_prf(df: pd.DataFrame, gold_col: str = "error_categories", pred_col: str = "rule_error_types") -> dict:
    """Per-category and micro precision/recall/F1 of the rules vs human labels (rows with a gold label and rule support only)."""
    rows = df[df[gold_col].notna() & (df.get("rule_supported", True) == True)]  # noqa: E712
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ERROR_TYPES}
    for _, r in rows.iterrows():
        gold, pred = parse_categories(r[gold_col]), parse_categories(r[pred_col])
        for t in ERROR_TYPES:
            stats[t]["tp"] += t in gold and t in pred
            stats[t]["fp"] += t not in gold and t in pred
            stats[t]["fn"] += t in gold and t not in pred

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "support": tp + fn}

    per = {t: prf(**s) for t, s in stats.items() if s["tp"] + s["fp"] + s["fn"] > 0}
    micro = prf(*(sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")))
    return {"n_rows": int(len(rows)), "per_category": per, "micro": micro}
