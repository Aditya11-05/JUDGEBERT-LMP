import pandas as pd
import re
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data/raw")

files = [
    base / "indian_laws.csv",
    base / "indian_laws_v2_breakdown.csv",
]

pairs = pd.read_csv(base / "final_dataset.csv").fillna("")
pairs = pairs[
    (pairs["legal_text"].astype(str).str.strip() != "") &
    (pairs["simplified_text"].astype(str).str.strip() != "")
].copy()

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

print("=" * 65)
print("INDIAN SOURCE DOCUMENT MATCHING")
print("=" * 65)

for file in files:
    if not file.exists():
        print(f"\nMissing: {file.name}")
        continue

    source = pd.read_csv(file).fillna("")
    print(f"\nSOURCE FILE: {file.name}")
    print("Rows:", len(source))
    print("Columns:", list(source.columns))

    for col in source.columns:
        print(f"\nColumn: {col}")
        for value in source[col].head(2):
            print(str(value)[:350].replace("\n", " "))

print("\n" + "=" * 65)
print("CHECKING INDIAN PAIRS AGAINST FULL SOURCE DOCUMENTS")
print("=" * 65)

full_path = base / "indian_laws.csv"
if full_path.exists():
    full = pd.read_csv(full_path).fillna("")

    for i, row in pairs.iterrows():
        original = normalize(row["legal_text"])
        if not original:
            continue

        for _, source_row in full.iterrows():
            for col in full.columns:
                document = normalize(source_row[col])
                if len(document) > 500 and original[:150] in document:
                    print(
                        f"Pair row {i + 1} matches source row "
                        f"{source_row.name + 1}, column {col}"
                    )
                    break
            else:
                continue
            break
