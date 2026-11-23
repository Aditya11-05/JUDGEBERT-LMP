#!/usr/bin/env python3
"""Create a controlled, synthetic LMP experiment dataset from legal text.

These labels describe known text transformations. They are not human legal
judgments and must not be presented as gold-standard legal annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


def split_into_chunks(text: str, max_words: int = 90) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    # Sentence-like chunks, retaining legal clause content as far as possible.
    parts = re.split(r"(?<=[.;:!?])\s+(?=[A-Z0-9(\"'])", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for part in parts:
        words = part.split()
        if not words:
            continue
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            for start in range(0, len(words), max_words):
                piece = " ".join(words[start:start + max_words])
                if len(piece.split()) >= 8:
                    chunks.append(piece)
            continue
        if current and count + len(words) > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(part)
        count += len(words)

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 8]


def stable_id(*parts: str) -> str:
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def transformations(text: str) -> list[tuple[str, str, float]]:
    """Return only explicit, reproducible changes with known edit type."""
    out: list[tuple[str, str, float]] = []

    patterns = [
        (r"\bshall\s+not\b", "shall", "negation_removal", 2.0),
        (r"\bmust\s+not\b", "must", "negation_removal", 2.0),
        (r"\bdoes\s+not\b", "does", "negation_removal", 2.5),
        (r"\bis\s+not\b", "is", "negation_removal", 2.5),
        (r"\bshall\b", "may", "obligation_to_permission", 3.0),
        (r"\bmust\b", "may", "obligation_to_permission", 3.0),
        (r"\bwill\b", "may", "certainty_to_permission", 4.0),
    ]

    for pattern, replacement, label, score in patterns:
        altered, n = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if n and altered != text:
            out.append((altered, label, score))

    # Alter an explicit integer only when it is not part of a year/date-like
    # token. This is a controlled numeric perturbation, not a legal rewrite.
    number_match = re.search(r"(?<![\w/.-])(\d{1,3})(?![\w/.-])", text)
    if number_match:
        old = int(number_match.group(1))
        new = old + 1 if old < 999 else old - 1
        altered = text[:number_match.start(1)] + str(new) + text[number_match.end(1):]
        if altered != text:
            out.append((altered, "numeric_value_change", 3.0))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/annotations/indian_lmp_annotation_candidates.csv",
    )
    parser.add_argument(
        "--output",
        default="datasets/experimental/lmp_controlled_synthetic.csv",
    )
    parser.add_argument("--max-words", type=int, default=90)
    args = parser.parse_args()

    source = pd.read_csv(args.input).fillna("")
    required = {"original_text", "source_document_id"}
    missing = required - set(source.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    seen: set[str] = set()

    for _, source_row in source.iterrows():
        doc_id = str(source_row.get("source_document_id", "")).strip()
        if not doc_id:
            doc_id = str(source_row.get("source_title", "unknown_document")).strip()
        if not doc_id:
            continue

        text = str(source_row.get("original_text", "")).strip()
        for chunk_no, chunk in enumerate(split_into_chunks(text, args.max_words)):
            # Positive identity control.
            candidates = [(chunk, "identity_control", 10.0)]
            candidates.extend(transformations(chunk))

            for altered, edit_type, rating in candidates:
                key = stable_id(doc_id, chunk, altered, edit_type)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "pair_id": key,
                    "source_document_id": doc_id,
                    "language": str(source_row.get("language", "en") or "en"),
                    "legal_domain": str(source_row.get("legal_domain", "") or ""),
                    "original_text": chunk,
                    "simplified_text": altered,
                    "experimental_rating": rating,
                    "transformation_type": edit_type,
                    "label_provenance": "controlled_synthetic_transformation",
                    "adjudication_status": "not_human_adjudicated",
                    "split_group_id": doc_id,
                    "chunk_index": chunk_no,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No usable chunks or transformation examples were generated.")

    # Deterministic split by document, never by individual pair.
    docs = sorted(result["source_document_id"].unique())
    doc_hashes = {
        d: int(hashlib.sha1(d.encode("utf-8")).hexdigest()[:8], 16) % 100
        for d in docs
    }
    result["split"] = result["source_document_id"].map(
        lambda d: "train" if doc_hashes[d] < 80
        else ("validation" if doc_hashes[d] < 90 else "test")
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(result):,}")
    print(f"Documents: {result.source_document_id.nunique():,}")
    print("\nTransformation counts:")
    print(result["transformation_type"].value_counts().to_string())
    print("\nSplit counts:")
    print(result["split"].value_counts().to_string())
    print("\nLabel distribution:")
    print(result["experimental_rating"].value_counts().sort_index().to_string())
    print("\nAll labels are controlled synthetic targets, not human judgments.")


if __name__ == "__main__":
    main()
