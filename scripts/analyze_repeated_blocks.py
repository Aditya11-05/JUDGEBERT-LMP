import pandas as pd
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

def normalize(text):
    return " ".join(str(text).lower().split())

df["original_norm"] = df["legal_text"].map(normalize)
df["simplified_norm"] = df["simplified_text"].map(normalize)

counts = df["original_norm"].value_counts()

print("DATASET SUMMARY")
print("Total records:", len(df))
print("Unique original passages:", df["original_norm"].nunique())
print("Repeated original passages:", (counts > 1).sum())
print("Original passages repeated 3+ times:", (counts >= 3).sum())
print("Exact duplicate pairs:", df.duplicated(
    ["original_norm", "simplified_norm"]
).sum())

variants = df.groupby("original_norm")["simplified_norm"].nunique()

print("Original passages with multiple simplifications:",
      (variants > 1).sum())

print("\nTOP 15 REPEATED ORIGINAL PASSAGES")

for text, count in counts[counts > 1].head(15).items():
    print(f"\nOccurrences: {count}")
    print(text[:250])

print("\nFIRST 30 RECORDS: PASSAGE GROUPS")

group_ids = {}
for i, text in enumerate(df["original_norm"].head(30), 1):
    if text not in group_ids:
        group_ids[text] = len(group_ids) + 1
    print(f"Record {i}: Passage group {group_ids[text]}")
