from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("datasets/raw/SIMPLE-LAW-source/outputs")

MAPPING_FILE = BASE / "indian_pair_document_mapping_v2.csv"
VALIDATION_FILE = BASE / "bnss_61_candidate_validation.csv"

OUTPUT_FILE = BASE / "indian_pair_document_mapping_v3.csv"
REPORT_FILE = BASE / "bnss_mapping_merge_report.txt"

BNSS_DOCUMENT_ID = "IN-ACT-15289dfc522f"
BNSS_DOCUMENT_TITLE = "Bharatiya Nagarik Suraksha Sanhita, 2023"

# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

for file in [MAPPING_FILE, VALIDATION_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Required file not found: {file}")

mapping = pd.read_csv(MAPPING_FILE)
validation = pd.read_csv(VALIDATION_FILE)

print("=" * 65)
print("JUDGEBERT - BNSS DOCUMENT MAPPING MERGER")
print("=" * 65)

print("\nExisting mapping records:", len(mapping))
print("Validated BNSS records:", len(validation))

# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

required_mapping = {
    "pair_row",
    "source_document_id",
    "source_document_title",
    "match_method",
    "match_status"
}

required_validation = {
    "pair_row",
    "source_document_id",
    "source_document_title",
    "bnss_section",
    "bnss_section_title",
    "validation_status",
    "validation_note"
}

missing_mapping = required_mapping - set(mapping.columns)
missing_validation = required_validation - set(validation.columns)

if missing_mapping:
    raise ValueError(
        f"Missing mapping columns: {sorted(missing_mapping)}"
    )

if missing_validation:
    raise ValueError(
        f"Missing validation columns: {sorted(missing_validation)}"
    )

# ============================================================
# 3. VALIDATE PAIR IDS
# ============================================================

mapping["pair_row"] = pd.to_numeric(
    mapping["pair_row"], errors="raise"
).astype(int)

validation["pair_row"] = pd.to_numeric(
    validation["pair_row"], errors="raise"
).astype(int)

if mapping["pair_row"].duplicated().any():
    duplicates = mapping.loc[
        mapping["pair_row"].duplicated(False), "pair_row"
    ].tolist()

    raise ValueError(f"Duplicate pair IDs in mapping: {duplicates}")

if validation["pair_row"].duplicated().any():
    duplicates = validation.loc[
        validation["pair_row"].duplicated(False), "pair_row"
    ].tolist()

    raise ValueError(f"Duplicate pair IDs in validation: {duplicates}")

if len(validation) != 61:
    raise ValueError(
        f"Expected 61 validated records, received {len(validation)}"
    )

if not validation["source_document_id"].eq(BNSS_DOCUMENT_ID).all():
    raise ValueError("Unexpected BNSS document ID in validation file.")

if not validation["source_document_title"].eq(
    BNSS_DOCUMENT_TITLE
).all():
    raise ValueError("Unexpected BNSS document title.")

if not validation["pair_row"].isin(mapping["pair_row"]).all():
    raise ValueError(
        "Some validated pair IDs do not exist in the original mapping."
    )

# ============================================================
# 4. CHECK EXISTING MAPPINGS FOR CONFLICTS
# ============================================================

existing = mapping.set_index("pair_row")
new_records = validation.set_index("pair_row")

conflicts = []

for pair_id, row in new_records.iterrows():
    old_id = existing.at[pair_id, "source_document_id"]

    if pd.notna(old_id) and str(old_id).strip():
        if str(old_id).strip() != BNSS_DOCUMENT_ID:
            conflicts.append({
                "pair_row": pair_id,
                "existing_document_id": old_id,
                "new_document_id": BNSS_DOCUMENT_ID
            })

if conflicts:
    raise ValueError(
        "Conflicting document IDs detected. No output written:\n"
        + str(conflicts)
    )

# ============================================================
# 5. ADD VALIDATION COLUMNS
# ============================================================

additional_columns = [
    "bnss_section",
    "bnss_section_title",
    "validation_status",
    "validation_note",
    "official_source"
]

for column in additional_columns:
    if column not in mapping.columns:
        mapping[column] = pd.NA

# ============================================================
# 6. MERGE THE 61 VALIDATED BNSS RECORDS
# ============================================================

validation = validation.reset_index(drop=True)

for _, row in validation.iterrows():

    pair_id = int(row["pair_row"])

    mask = mapping["pair_row"] == pair_id

    mapping.loc[mask, "source_document_id"] = BNSS_DOCUMENT_ID

    mapping.loc[mask, "source_document_title"] = (
        BNSS_DOCUMENT_TITLE
    )

    mapping.loc[mask, "match_method"] = (
        "official_bnss_text_match"
    )

    mapping.loc[mask, "match_status"] = (
        "confirmed_external"
    )

    mapping.loc[mask, "bnss_section"] = row["bnss_section"]

    mapping.loc[mask, "bnss_section_title"] = (
        row["bnss_section_title"]
    )

    mapping.loc[mask, "validation_status"] = (
        row["validation_status"]
    )

    mapping.loc[mask, "validation_note"] = (
        row["validation_note"]
    )

    if "official_source" in validation.columns:
        mapping.loc[mask, "official_source"] = (
            row["official_source"]
        )

# ============================================================
# 7. FINAL INTEGRITY CHECKS
# ============================================================

if len(mapping) != 2126:
    raise ValueError(
        f"Unexpected final row count: {len(mapping)}"
    )

if mapping["pair_row"].duplicated().any():
    raise ValueError("Duplicate pair IDs after merging.")

bnss_rows = mapping[
    mapping["source_document_id"] == BNSS_DOCUMENT_ID
]

if len(bnss_rows) < 61:
    raise ValueError(
        f"Expected at least 61 BNSS mappings, found {len(bnss_rows)}"
    )

validated_rows = mapping[
    mapping["validation_status"].notna()
]

if len(validated_rows) != 61:
    raise ValueError(
        f"Expected 61 validation records, found {len(validated_rows)}"
    )

# ============================================================
# 8. SAVE NEW MAPPING
# ============================================================

mapping = mapping.sort_values("pair_row").reset_index(drop=True)

mapping.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 9. GENERATE MERGE REPORT
# ============================================================

mapped_count = mapping["source_document_id"].notna().sum()
unmapped_count = len(mapping) - mapped_count

report = f"""
JUDGEBERT - BNSS MAPPING MERGE REPORT
=====================================

Original mapping file:
{MAPPING_FILE}

Validation file:
{VALIDATION_FILE}

Output file:
{OUTPUT_FILE}

TOTAL RECORDS
-------------
Original records: {len(mapping)}
Validated BNSS records merged: {len(validation)}
Total mapped records: {mapped_count}
Total unmapped records: {unmapped_count}

BNSS VALIDATION
---------------
BNSS document ID: {BNSS_DOCUMENT_ID}
BNSS document title: {BNSS_DOCUMENT_TITLE}
BNSS records in final mapping: {len(bnss_rows)}
Validated records: {len(validated_rows)}

Validation statuses:
{validated_rows["validation_status"].value_counts().to_string()}

INTEGRITY
---------
Duplicate pair IDs: {mapping["pair_row"].duplicated().sum()}
Total records preserved: {len(mapping) == 2126}
Original mapping file overwritten: No

STATUS: SUCCESS
"""

REPORT_FILE.write_text(report, encoding="utf-8")

print(report)
print("Saved mapping:", OUTPUT_FILE)
print("Saved report:", REPORT_FILE)

print("\n" + "=" * 65)
print("BNSS MAPPING MERGE COMPLETED SUCCESSFULLY")
print("=" * 65)
