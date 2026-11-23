import csv

import pytest

from app.data import annotation_store as store


@pytest.fixture
def candidate_csv(tmp_path):
    path = tmp_path / "candidates.csv"
    rows = [
        {
            "pair_id": "TEST-001",
            "source_document_id": "DOC-1",
            "source_title": "Synthetic test document",
            "source_licence": "UNVERIFIED",
            "language": "en",
            "original_text": "The tenant must pay by the fifth day.",
            "simplified_text": "The tenant may pay by the fifth day.",
            "simplification_origin": "unknown",
        },
        {
            "pair_id": "TEST-002",
            "source_document_id": "DOC-2",
            "source_title": "Synthetic test document",
            "source_licence": "UNVERIFIED",
            "language": "en",
            "original_text": "A notice must be given.",
            "simplified_text": "A notice should be given.",
            "simplification_origin": "unknown",
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return path


@pytest.fixture
def database(tmp_path, candidate_csv):
    path = tmp_path / "annotation_test.sqlite3"
    result = store.import_candidate_csv(candidate_csv, path)
    assert result == {"inserted": 2, "skipped": 0}
    return path


def submit(
    db,
    pair="TEST-001",
    reviewer="REVIEWER-A",
    rating=5,
    qualification="CS undergraduate, no legal training",
):
    return store.submit_annotation(
        pair_id=pair,
        annotator_id=reviewer,
        qualification=qualification,
        rating=rating,
        confidence=4,
        error_categories=["obligation_permission_change"],
        annotation_notes="Test annotation.",
        db_path=db,
    )


def test_import_is_idempotent(database, candidate_csv):
    result = store.import_candidate_csv(candidate_csv, database)

    assert result == {"inserted": 0, "skipped": 2}
    assert store.get_candidate_count(database) == 2


def test_independent_reviewers_have_separate_queues(database):
    submit(database, reviewer="REVIEWER-A")

    assert store.get_next_pair("REVIEWER-A", database)["pair_id"] == "TEST-002"
    assert store.get_next_pair("REVIEWER-B", database)["pair_id"] == "TEST-001"


def test_duplicate_submission_is_rejected(database):
    submit(database)

    with pytest.raises(ValueError, match="already annotated"):
        submit(database, rating=6)


def test_invalid_rating_and_confidence_are_rejected(database):
    with pytest.raises(ValueError, match="Rating"):
        submit(database, rating=11)

    with pytest.raises(ValueError, match="Confidence"):
        store.submit_annotation(
            pair_id="TEST-001",
            annotator_id="REVIEWER-B",
            qualification="Law student",
            rating=7,
            confidence=0,
            error_categories=[],
            db_path=database,
        )


def test_unknown_error_category_is_rejected(database):
    with pytest.raises(ValueError, match="Unknown error categories"):
        store.submit_annotation(
            pair_id="TEST-001",
            annotator_id="REVIEWER-A",
            qualification="CS undergraduate",
            rating=7,
            confidence=4,
            error_categories=["invented_category"],
            db_path=database,
        )


def test_adjudication_requires_two_independent_reviewers(database):
    submit(database)

    with pytest.raises(ValueError, match="two independent annotations"):
        store.record_adjudication(
            pair_id="TEST-001",
            adjudicator_id="ADJUDICATOR",
            adjudicated_rating=5,
            resolution_notes="Test resolution.",
            db_path=database,
        )


def test_adjudication_records_disagreement_spread(database):
    submit(database, reviewer="REVIEWER-A", rating=5)
    submit(
        database,
        reviewer="REVIEWER-B",
        rating=8,
        qualification="Law student",
    )

    inputs = store.get_adjudication_inputs("TEST-001", database)
    assert len(inputs) == 2
    assert {item["rating"] for item in inputs} == {5, 8}

    result_id = store.record_adjudication(
        pair_id="TEST-001",
        adjudicator_id="ADJUDICATOR",
        adjudicated_rating=6,
        resolution_notes="Reviewers discussed the difference.",
        db_path=database,
    )

    assert result_id > 0

    with pytest.raises(ValueError, match="already exists"):
        store.record_adjudication(
            pair_id="TEST-001",
            adjudicator_id="ADJUDICATOR",
            adjudicated_rating=6,
            resolution_notes="Duplicate test.",
            db_path=database,
        )


def test_summary_counts(database):
    submit(database)

    assert store.get_annotation_summary(database) == {
        "candidate_pairs": 2,
        "annotations": 1,
        "annotators": 1,
        "adjudications": 0,
    }


def test_adjudication_queue_requires_two_reviewers(database):
    submit(database, reviewer="REVIEWER-A", rating=5)

    assert store.get_pairs_pending_adjudication(database) == []

    submit(
        database,
        reviewer="REVIEWER-B",
        rating=8,
        qualification="Law student",
    )

    pending = store.get_pairs_pending_adjudication(database)

    assert len(pending) == 1
    assert pending[0]["pair_id"] == "TEST-001"
    assert pending[0]["reviewer_count"] == 2
    assert pending[0]["minimum_rating"] == 5
    assert pending[0]["maximum_rating"] == 8


def test_adjudicated_pair_leaves_pending_queue(database):
    submit(database, reviewer="REVIEWER-A", rating=5)
    submit(
        database,
        reviewer="REVIEWER-B",
        rating=8,
        qualification="Law student",
    )

    store.record_adjudication(
        pair_id="TEST-001",
        adjudicator_id="ADJUDICATOR",
        adjudicated_rating=6,
        resolution_notes="Test resolution.",
        db_path=database,
    )

    assert store.get_pairs_pending_adjudication(database) == []
