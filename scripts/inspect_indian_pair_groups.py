import pandas as pd
import re
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

original = df["legal_text"].astype(str).str.strip()
simplified = df["simplified_text"].astype(str).str.strip()

valid = (original != "") & (simplified != "")
pairs = df[valid].copy()
pairs["row_number"] = pairs.index + 1

public_law = re.compile(r"^\s*\*?\s*Public Law \d+-\d+", re.I)
indian = pairs[
    ~pairs["legal_text"].str.contains(public_law, na=False)
]

print("NON-PUBLIC-LAW PAIRS:", len(indian))
print("\nSAMPLE RECORDS ACROSS THE DATASET")

for _, row in indian.iloc[::50].iterrows():
    text = str(row["legal_text"]).replace("\n", " ")
    print(f"\nRow {row['row_number']}: {text[:220]}")

print("\nFIRST 30 NON-PUBLIC-LAW RECORDS")
for _, row in indian.head(30).iterrows():
    text = str(row["legal_text"]).replace("\n", " ")
    print(f"\nRow {row['row_number']}: {text[:180]}")
