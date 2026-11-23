"""SQLite storage for independent human LMP annotations.

This module stores annotations separately from the candidate CSV.
It does not calculate LMP scores, create labels, or determine training eligibility.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DEFAULT_DB_PATH = Path("datasets/annotations/annotation_studio.sqlite3")

ERROR_CATEGORIES = {
    "obligation_permission_change",
    "negation_change",
    "date_or_deadline_change",
    "bound_or_direction_change",
    "monetary_amount_change",
    "percentage_change",
    "numerical_value_change",
    "frequency_change",
    "party_or_entity_change",
    "scope_change",
    "condition_removed",
    "condition_added",
    "condition_strength_change",
    "exception_removed",
    "exception_added",
    "possible_omission_or_addition",
    "contradiction",
    "hallucination",
    "other",
}

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS candidate_pairs (
    pair_id TEXT PRIMARY KEY,
    source_document_id TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL DEFAULT '',
    source_licence TEXT NOT NULL DEFAULT 'UNVERIFIED',
    language TEXT NOT NULL DEFAULT '',
    legal_domain TEXT NOT NULL DEFAULT '',
    original_text TEXT NOT NULL,
    simplified_text TEXT NOT NULL,
    simplification_origin TEXT NOT NULL DEFAULT 'unknown',
    split_group_id TEXT NOT NULL DEFAULT '',
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotators (
    annotator_id TEXT PRIMARY KEY,
    qualification TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id TEXT NOT NULL REFERENCES candidate_pairs(pair_id),
    annotator_id TEXT NOT NULL REFERENCES annotators(annotator_id),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 10),
    confidence INTEGER NOT NULL CHECK (confidence BETWEEN 1 AND 5),
    error_categories TEXT NOT NULL,
    annotation_notes TEXT NOT NULL DEFAULT '',
    submitted_at TEXT NOT NULL,
    UNIQUE(pair_id, annotator_id)
);

CREATE TABLE IF NOT EXISTS adjudications (
    adjudication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id TEXT NOT NULL UNIQUE REFERENCES candidate_pairs(pair_id),
    adjudicator_id TEXT NOT NULL,
    adjudicated_rating INTEGER NOT NULL
        CHECK (adjudicated_rating BETWEEN 1 AND 10),
    disagreement_spread INTEGER NOT NULL CHECK (disagreement_spread >= 0),
    resolution_notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotations_pair
    ON annotations(pair_id);

CREATE INDEX IF NOT EXISTS idx_annotations_reviewer
    ON annotations(annotator_id);

CREATE INDEX IF NOT EXISTS idx_pairs_document
    ON candidate_pairs(source_document_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Create the database schema without deleting existing records."""
    with _connection(db_path) as conn:
        conn.executescript(SCHEMA)


def import_candidate_csv(
    csv_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Import candidate pairs once; never overwrite existing pairs or annotations."""
    initialize_database(db_path)

    inserted = 0
    skipped = 0
    seen_ids: set[str] = set()

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"pair_id", "original_text", "simplified_text"}
        available = set(reader.fieldnames or [])
        missing = required - available
        if missing:
            raise ValueError(
                f"Candidate CSV is missing required columns: {sorted(missing)}"
            )

        with _connection(db_path) as conn:
            for row in reader:
                pair_id = (row.get("pair_id") or "").strip()
                original = (row.get("original_text") or "").strip()
                simplified = (row.get("simplified_text") or "").strip()

                if not pair_id or not original or not simplified:
                    skipped += 1
                    continue

                if pair_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(pair_id)

                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO candidate_pairs (
                        pair_id, source_document_id, source_url,
                        source_title, source_licence, language,
                        legal_domain, original_text, simplified_text,
                        simplification_origin, split_group_id, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pair_id,
                        (row.get("source_document_id") or "").strip(),
                        (row.get("source_url") or "").strip(),
                        (
                            row.get("source_title")
                            or row.get("source_document_title")
                            or ""
                        ).strip(),
                        (row.get("source_licence") or "UNVERIFIED").strip(),
                        (row.get("language") or "").strip(),
                        (row.get("legal_domain") or "").strip(),
                        original,
                        simplified,
                        (row.get("simplification_origin") or "unknown").strip(),
                        (
                            row.get("split_group_id")
                            or row.get("source_document_id")
                            or ""
                        ).strip(),
                        _now(),
                    ),
                )

                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1

    return {"inserted": inserted, "skipped": skipped}


def get_candidate_count(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM candidate_pairs"
        ).fetchone()
        return int(row["total"])


def get_pair(
    pair_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM candidate_pairs WHERE pair_id = ?",
            (pair_id,),
        ).fetchone()
        return dict(row) if row else None


def get_next_pair(
    annotator_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """Return the next pair not yet annotated by this reviewer."""
    reviewer = annotator_id.strip()
    if not reviewer:
        raise ValueError("Annotator ID cannot be empty.")

    with _connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.*
            FROM candidate_pairs AS p
            WHERE NOT EXISTS (
                SELECT 1
                FROM annotations AS a
                WHERE a.pair_id = p.pair_id
                  AND a.annotator_id = ?
            )
            ORDER BY p.pair_id
            LIMIT 1
            """,
            (reviewer,),
        ).fetchone()

        return dict(row) if row else None


def submit_annotation(
    *,
    pair_id: str,
    annotator_id: str,
    qualification: str,
    rating: int,
    confidence: int,
    error_categories: list[str],
    annotation_notes: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Save one independent, immutable annotation."""
    pair_id = pair_id.strip()
    annotator_id = annotator_id.strip()
    qualification = qualification.strip()

    if not pair_id:
        raise ValueError("Pair ID cannot be empty.")
    if not annotator_id:
        raise ValueError("Annotator ID cannot be empty.")
    if not qualification:
        raise ValueError("Annotator qualification must be declared.")
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 10:
        raise ValueError("Rating must be an integer from 1 to 10.")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 1 <= confidence <= 5
    ):
        raise ValueError("Confidence must be an integer from 1 to 5.")
    if not isinstance(error_categories, list):
        raise ValueError("Error categories must be a list.")

    categories = sorted(set(error_categories))
    invalid = set(categories) - ERROR_CATEGORIES
    if invalid:
        raise ValueError(f"Unknown error categories: {sorted(invalid)}")

    if not isinstance(annotation_notes, str):
        raise ValueError("Annotation notes must be text.")

    with _connection(db_path) as conn:
        pair = conn.execute(
            "SELECT 1 FROM candidate_pairs WHERE pair_id = ?",
            (pair_id,),
        ).fetchone()
        if pair is None:
            raise ValueError(f"Unknown candidate pair: {pair_id}")

        existing_reviewer = conn.execute(
            "SELECT qualification FROM annotators WHERE annotator_id = ?",
            (annotator_id,),
        ).fetchone()

        if existing_reviewer is not None:
            if existing_reviewer["qualification"] != qualification:
                raise ValueError(
                    "This reviewer ID already has a different qualification "
                    "record. Do not silently change reviewer credentials."
                )
        else:
            conn.execute(
                """
                INSERT INTO annotators (annotator_id, qualification, created_at)
                VALUES (?, ?, ?)
                """,
                (annotator_id, qualification, _now()),
            )

        try:
            cursor = conn.execute(
                """
                INSERT INTO annotations (
                    pair_id, annotator_id, rating, confidence,
                    error_categories, annotation_notes, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_id,
                    annotator_id,
                    rating,
                    confidence,
                    json.dumps(categories, ensure_ascii=False),
                    annotation_notes.strip(),
                    _now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "This reviewer has already annotated this pair. "
                "A revision workflow has not yet been enabled."
            ) from exc

        return int(cursor.lastrowid)


def get_adjudication_inputs(
    pair_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Retrieve ratings for the adjudication stage, not the blind annotation UI."""
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.annotator_id, n.rating, n.confidence,
                   n.error_categories, n.annotation_notes, n.submitted_at
            FROM annotations AS n
            JOIN annotators AS a ON a.annotator_id = n.annotator_id
            WHERE n.pair_id = ?
            ORDER BY n.submitted_at, n.annotation_id
            """,
            (pair_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def record_adjudication(
    *,
    pair_id: str,
    adjudicator_id: str,
    adjudicated_rating: int,
    resolution_notes: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Record an adjudicated rating after at least two independent submissions."""
    pair_id = pair_id.strip()
    adjudicator_id = adjudicator_id.strip()
    resolution_notes = resolution_notes.strip()

    if not pair_id or not adjudicator_id:
        raise ValueError("Pair ID and adjudicator ID are required.")
    if (
        isinstance(adjudicated_rating, bool)
        or not isinstance(adjudicated_rating, int)
        or not 1 <= adjudicated_rating <= 10
    ):
        raise ValueError("Adjudicated rating must be an integer from 1 to 10.")
    if not resolution_notes:
        raise ValueError("Adjudication resolution notes are required.")

    with _connection(db_path) as conn:
        ratings = conn.execute(
            """
            SELECT rating
            FROM annotations
            WHERE pair_id = ?
            ORDER BY submitted_at, annotation_id
            """,
            (pair_id,),
        ).fetchall()

        values = [int(row["rating"]) for row in ratings]
        if len(values) < 2:
            raise ValueError(
                "At least two independent annotations are required."
            )

        spread = max(values) - min(values)

        if spread > 2 and not resolution_notes:
            raise ValueError(
                "A disagreement greater than two points requires resolution notes."
            )

        try:
            cursor = conn.execute(
                """
                INSERT INTO adjudications (
                    pair_id, adjudicator_id, adjudicated_rating,
                    disagreement_spread, resolution_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_id,
                    adjudicator_id,
                    adjudicated_rating,
                    spread,
                    resolution_notes,
                    _now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "An adjudication already exists for this pair."
            ) from exc

        return int(cursor.lastrowid)


def get_annotation_summary(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    with _connection(db_path) as conn:
        pairs = conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_pairs"
        ).fetchone()["n"]
        annotations = conn.execute(
            "SELECT COUNT(*) AS n FROM annotations"
        ).fetchone()["n"]
        reviewers = conn.execute(
            "SELECT COUNT(*) AS n FROM annotators"
        ).fetchone()["n"]
        adjudications = conn.execute(
            "SELECT COUNT(*) AS n FROM adjudications"
        ).fetchone()["n"]

    return {
        "candidate_pairs": int(pairs),
        "annotations": int(annotations),
        "annotators": int(reviewers),
        "adjudications": int(adjudications),
    }


def get_pairs_pending_adjudication(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Return pairs with at least two independent ratings and no adjudication."""
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                p.pair_id,
                p.source_document_id,
                p.original_text,
                p.simplified_text,
                COUNT(DISTINCT n.annotator_id) AS reviewer_count,
                MIN(n.rating) AS minimum_rating,
                MAX(n.rating) AS maximum_rating
            FROM candidate_pairs AS p
            JOIN annotations AS n ON n.pair_id = p.pair_id
            LEFT JOIN adjudications AS j ON j.pair_id = p.pair_id
            WHERE j.pair_id IS NULL
            GROUP BY p.pair_id
            HAVING COUNT(DISTINCT n.annotator_id) >= 2
            ORDER BY p.pair_id
            """
        ).fetchall()

        return [dict(row) for row in rows]
