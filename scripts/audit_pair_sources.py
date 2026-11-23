import pandas as pd
import re
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

df["row_number"] = range(1, len(df) + 1)

original = df["legal_text"].astype(str).str.strip()
simplified = df["simplified_text"].astype(str).str.strip()

valid = (original != "") & (simplified != "")
pairs = df[valid].copy()

public_law = re.compile(
    r"^\s*\*?\s*(Public Law \d+-\d+)",
    re.IGNORECASE
)

pairs["public_law_id"] = pairs["legal_text"].apply(
    lambda x: (m.group(1).upper() if (m := public_law.search(str(x))) else None)
)

print("=" * 60)
print("SOURCE STRUCTURE OF USABLE PAIRS")
print("=" * 60)

print("Total usable pairs:", len(pairs))
print("Pairs with Public Law header:",
      pairs["public_law_id"].notna().sum())
print("Unique Public Law IDs:",
      pairs["public_law_id"].nunique())
print("Pairs without Public Law header:",
      pairs["public_law_id"].isna().sum())

print("\nPAIRS BY ROW RANGE")

for start, end in [(1, 100), (101, 500), (501, 1000),
                   (1001, 1333), (1334, 2126)]:
    subset = pairs[
        pairs["row_number"].between(start, end)
    ]
    print(f"Rows {start}-{end}: {len(subset)} usable pairs")

print("\nFIRST 15 NON-PUBLIC-LAW PAIRS AFTER ROW 1000")

subset = pairs[
    (pairs["row_number"] > 1000) &
    (pairs["public_law_id"].isna())
]

for _, row in subset.head(15).iterrows():
    print(f"\nRow {row['row_number']}")
    print(str(row["legal_text"])[:250].replace("\n", " "))

print("\nPUBLIC LAW ID EXAMPLES")
for value in pairs["public_law_id"].dropna().drop_duplicates().head(15):
    print(value)
