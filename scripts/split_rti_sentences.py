import re
import csv
import pandas as pd
from pathlib import Path

source = Path("datasets/processed/rti_all_section_candidates.csv")
output = Path("datasets/processed/rti_sentence_candidates.csv")

df = pd.read_csv(source, encoding="utf-8-sig")
rows = []

def split_legal_text(text):
    # Split after sentence-ending punctuation or semicolons.
    # Semicolon-separated legal clauses remain independently reviewable.
    parts = re.split(r'(?<=[.;])\s+(?=(?:\(?[a-z0-9ivx]+\)?[.)]|[A-Z“"(]))', str(text))

    return [p.strip() for p in parts if len(p.strip()) >= 25]

for _, section in df.iterrows():
    units = split_legal_text(section["original_section_text"])

    for unit in units:
        rows.append({
            "pair_id": f"RTI-EN-{len(rows)+1:04d}",
            "source_document_id": "RTI_ACT_2005",
            "section_number": int(section["section_number"]),
            "section_title": section["section_title"],
            "language": "en",
            "original_text": unit,
            "simplified_text": "",
            "simplification_origin": "",
            "annotation_rating": "",
            "annotation_confidence": "",
            "review_status": "needs_manual_review"
        })

result = pd.DataFrame(rows)
result.to_csv(output, index=False, encoding="utf-8-sig")

print("Sections processed:", df["section_number"].nunique())
print("Sentence/clause candidates:", len(result))
print("Output:", output)
print("\nCandidates by section:")
print(result.groupby("section_number").size().to_string())
