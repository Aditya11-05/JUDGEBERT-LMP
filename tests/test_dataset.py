import pandas as pd
import pytest
from app.data.validate_dataset import (
    validate, validate_training_data, find_duplicates, drop_duplicates
)
from app.data.split_dataset import group_split, assert_no_leakage


def make(n=40):
    return pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(n)],
        "source_document_id": [f"doc{i % 10}" for i in range(n)],
        "language": ["en"] * n, "legal_domain": ["test"] * n,
        "original_text": [f"Clause number {i} says the party shall act." for i in range(n)],
        "simplified_text": [f"Item {i}: the party must act." for i in range(n)],
        "simplification_origin": ["synthetic"] * n,
        "source_licence": ["test-fixture"] * n,
        "annotator_rating": [5.0] * n,
    })


def test_valid():
    assert validate(make()) == []


def test_catches_problems():
    df = make()
    df.loc[0, "language"] = "xx"
    df.loc[1, "annotator_rating"] = 11
    df.loc[2, "original_text"] = " "
    errs = " ".join(validate(df))
    assert "language" in errs and "annotator_rating" in errs and "original_text" in errs
    assert "missing columns" in validate(df.drop(columns=["language"]))[0]


def test_duplicates():
    df = make(5)
    df.loc[4] = df.loc[0].copy()
    df.loc[4, "pair_id"] = "dup"
    assert ("p0", "dup", 1.0) in find_duplicates(df)
    assert "dup" not in drop_duplicates(df)["pair_id"].tolist()


def test_split_is_grouped_reproducible():
    df = make(100)
    a, b = group_split(df, seed=1), group_split(df, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert_no_leakage(a)
    assert set(a["split"]) == {"train", "val", "test"}
    assert group_split(df, seed=2)["split"].tolist() != a["split"].tolist()


def test_leakage_detected():
    df = group_split(make(100))
    df.loc[df.index[0], "split"] = "test" if df.loc[df.index[0], "split"] != "test" else "train"
    with pytest.raises(AssertionError):
        assert_no_leakage(df)


def make_training_data():
    df = make(40)
    df["source_document_id"] = [f"verified_doc_{i % 10}" for i in range(40)]
    df["simplification_origin"] = "human_written"
    df["source_licence"] = "CC-BY-4.0"
    df["licence_status"] = "verified"
    df["adjudicated_rating"] = 8.0
    df["adjudication_status"] = "adjudicated"
    return df


def test_strict_training_data_valid():
    assert validate_training_data(make_training_data()) == []


def test_strict_training_rejects_unresolved_documents():
    df = make_training_data()
    df.loc[0, "source_document_id"] = "UNRESOLVED"
    assert any("unresolved" in e for e in validate_training_data(df))


def test_strict_training_rejects_unverified_licence():
    df = make_training_data()
    df.loc[0, "licence_status"] = "unverified"
    assert any("licence_status" in e for e in validate_training_data(df))


def test_strict_training_rejects_synthetic_data():
    df = make_training_data()
    df.loc[0, "simplification_origin"] = "synthetic"
    assert any("synthetic" in e for e in validate_training_data(df))


def test_strict_training_rejects_missing_adjudicated_rating():
    df = make_training_data()
    df.loc[0, "adjudicated_rating"] = None
    assert any("adjudicated_rating" in e for e in validate_training_data(df))


def test_strict_training_rejects_unadjudicated_rating():
    df = make_training_data()
    df.loc[0, "adjudication_status"] = "pending"
    assert any("adjudication_status" in e for e in validate_training_data(df))
