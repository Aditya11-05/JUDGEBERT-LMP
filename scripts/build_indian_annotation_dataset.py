import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/raw/SIMPLE-LAW-source"

source = pd.read_csv(
    BASE / "data/processed/INDIAN_LAW_DATASET.csv"
).fillna("")

mapping = pd.read_csv(
    BASE / "outputs/indian_document_id_mapping.csv"
)

if len(source) != len(mapping):
    raise SystemExit("ERROR: Dataset and mapping row counts differ.")

if mapping["source_row"].tolist() != list(range(1, len(source) + 1)):
    raise SystemExit("ERROR: Source row alignment failed.")

if mapping["source_document_id"].isna().any():
    raise SystemExit("ERROR: Missing document IDs.")

df = pd.DataFrame({
    "pair_id": [f"IN-LMP-{i:05d}" for i in range(1, len(source) + 1)],
    "source_document_id": mapping["source_document_id"],
    "source_document_title": mapping["source_document_title"],
    "mapping_method": mapping["document_id_method"],
    "mapping_status": mapping["document_id_status"],
    "language": "en",
    "language_verification_status": "pending",
    "legal_domain": "Indian law - domain review required",
    "original_text": source["legal_text"],
    "simplified_text": source["simplified_text"],
    "simplification_origin": "unknown",
    "source_licence": "UNVERIFIED",
    "licence_status": "unverified",
    "annotator_id": "",
    "annotator_rating": pd.NA,
    "adjudicated_rating": pd.NA,
    "adjudication_status": "pending",
    "confidence": "",
    "error_categories": "",
    "annotation_notes": "",
    "source_url": "",
    "source_passage_group_id": "",
    "split_group_id": mapping["source_document_id"],
})

output = ROOT / "datasets/annotations/indian_lmp_annotation_candidates.csv"
df.to_csv(output, index=False)

print("CREATED:", output)
print("ROWS:", len(df))
print("UNIQUE DOCUMENTS:", df["source_document_id"].nunique())
print("ANNOTATION STATUS:", df["adjudication_status"].value_counts().to_dict())
print("LICENCE STATUS:", df["licence_status"].value_counts().to_dict())
print("RATINGS CREATED: 0")
