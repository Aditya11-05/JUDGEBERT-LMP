import pandas as pd
import re
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data")

pairs = pd.read_csv(base / "raw/final_dataset.csv").fillna("")
pairs = pairs[
    (pairs["legal_text"].astype(str).str.strip() != "") &
    (pairs["simplified_text"].astype(str).str.strip() != "")
]

def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

sources = [
    "raw/indian_laws.csv",
    "raw/indian_laws_v2_breakdown.csv",
    "processed/INDIAN_LAW_DATASET.csv",
]

for filename in sources:
    path = base / filename
    if not path.exists():
        continue

    source = pd.read_csv(path).fillna("")
    documents = [
        norm(x) for x in source["legal_text"]
        if len(norm(x)) > 100
    ]

    print("\n" + "=" * 60)
    print("SOURCE:", filename)
    print("Source records:", len(documents))

    for prefix_length in [80, 120, 150, 200]:
        matches = 0

        for text in pairs["legal_text"]:
            prefix = norm(text)[:prefix_length]
            if len(prefix) < prefix_length:
                continue

            if any(prefix in document for document in documents):
                matches += 1

        print(f"Prefix {prefix_length}: {matches}/{len(pairs)} matches")

print("\nSAMPLE SOURCE DOCUMENT HEADERS")
source = pd.read_csv(base / "raw/indian_laws.csv").fillna("")
for i, value in enumerate(source["legal_text"].head(10)):
    print(f"\nRecord {i + 1}: {str(value)[:300]}")
