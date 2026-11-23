"""Plumbing tests for dataset integration. The rows below are TEST FIXTURES, not research data or results."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from app.data.validate_dataset import unknown_error_categories, validate
from app.explanations.dataset import annotate_dataframe, category_prf, parse_categories

ROOT = Path(__file__).resolve().parents[1]


def fixture_df():
    rows = [
        ("t1", "en", "The tenant must pay rent.", "The tenant may pay rent.", "obligation_permission_change"),
        ("t2", "en", "Pay Rs. 500 by 5 May 2026.", "Pay Rs. 50 by 5 May 2026.", "monetary_amount_change"),
        ("t3", "en", "The tenant must pay rent.", "The tenant is required to pay rent.", ""),
        ("t4", "en", "Pay rent unless the landlord waives it.", "Pay rent.", "exception_removed;condition_removed"),
        ("t5", "hi", "किरायेदार को किराया देना होगा।", "किरायेदार किराया दे सकता है।", "obligation_permission_change"),
    ]
    return pd.DataFrame({
        "pair_id": [r[0] for r in rows], "source_document_id": [f"d{i}" for i in range(len(rows))],
        "language": [r[1] for r in rows], "legal_domain": ["fixture"] * len(rows),
        "original_text": [r[2] for r in rows], "simplified_text": [r[3] for r in rows],
        "simplification_origin": ["synthetic"] * len(rows), "source_licence": ["test-fixture"] * len(rows),
        "error_categories": [r[4] or None for r in rows],
    })


def test_annotate_adds_columns_and_leaves_input_untouched():
    df = fixture_df()
    before = df.copy()
    out = annotate_dataframe(df)
    pd.testing.assert_frame_equal(df, before)
    assert {"rule_supported", "rule_error_types", "rule_n_findings", "rule_findings"} <= set(out.columns)
    assert "obligation_permission_change" in out.loc[0, "rule_error_types"].split(";")
    assert out.loc[2, "rule_n_findings"] == 0 and out.loc[2, "rule_error_types"] == ""
    assert isinstance(json.loads(out.loc[0, "rule_findings"]), list)


def test_hindi_rows_are_skipped_not_faked():
    out = annotate_dataframe(fixture_df())
    row = out[out["pair_id"] == "t5"].iloc[0]
    assert not bool(row["rule_supported"])
    assert pd.isna(row["rule_error_types"]) and pd.isna(row["rule_findings"]) and pd.isna(row["rule_n_findings"])


def test_output_format_matches_error_categories_column():
    out = annotate_dataframe(fixture_df())
    assert parse_categories(out.loc[3, "rule_error_types"]) == {"exception_removed", "party_or_entity_change"}  # landlord clause also dropped; same ';' format as gold
    assert parse_categories("a; b;none") == {"a", "b"} and parse_categories(None) == set()


def test_category_prf_counts_and_excludes_unsupported_rows():
    res = category_prf(annotate_dataframe(fixture_df()))
    assert res["n_rows"] == 3  # t1, t2, t4 have gold labels and rule support; t3 has no gold; t5 is Hindi
    per = res["per_category"]
    assert per["obligation_permission_change"]["tp"] == 1 and per["monetary_amount_change"]["tp"] == 1
    assert per["exception_removed"]["tp"] == 1
    assert per["party_or_entity_change"]["fp"] == 1  # rules flagged the dropped landlord; the fixture's gold label omits it
    assert per["condition_removed"] == {"tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0.0, "f1": None, "support": 1}
    assert 0.0 <= res["micro"]["precision"] <= 1.0 and 0.0 <= res["micro"]["recall"] <= 1.0


def test_category_prf_handles_empty_denominators():
    df = annotate_dataframe(fixture_df()).iloc[[2]]  # no gold label -> no rows scored
    res = category_prf(df)
    assert res["n_rows"] == 0 and res["micro"]["precision"] is None and res["per_category"] == {}


def test_vocabulary_check_flags_unknown_labels():
    df = fixture_df()
    df.loc[0, "error_categories"] = "obligation_permission_change;made_up_label"
    assert unknown_error_categories(df) == {"made_up_label": [0]}
    assert unknown_error_categories(fixture_df()) == {}
    assert validate(fixture_df()) == []  # fixture is schema-valid; vocabulary check is separate and opt-in


def test_cli_json_and_error_codes():
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / "explain_pair.py"), *a],
                                    capture_output=True, text=True, cwd=ROOT)
    ok = run("The tenant must pay rent.", "The tenant may pay rent.", "--json")
    assert ok.returncode == 0 and json.loads(ok.stdout)["findings"][0]["error_type"] == "obligation_permission_change"
    assert run("a", "b", "--language", "hi").returncode == 2
    assert run("", "b").returncode == 2
