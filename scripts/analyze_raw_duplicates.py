import pandas as pd
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data")

raw = pd.read_csv(base / "raw/final_dataset.csv")
processed = pd.read_csv(base / "processed/simplification_final_dataset.csv")

def normalize(series):
    return series.fillna("").astype(str).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

for df in [raw, processed]:
    df["legal_norm"] = normalize(df["legal_text"])
    df["simple_norm"] = normalize(df["simplified_text"])

print("RAW DATASET")
print("Rows:", len(raw))
print("Unique original texts:", raw["legal_norm"].nunique())
print("Unique exact pairs:", raw[["legal_norm", "simple_norm"]].drop_duplicates().shape[0])
print("Duplicate exact pairs:", raw.duplicated(["legal_norm", "simple_norm"]).sum())

print("\nPROCESSED DATASET")
print("Rows:", len(processed))
print("Unique original texts:", processed["legal_norm"].nunique())
print("Unique exact pairs:", processed[["legal_norm", "simple_norm"]].drop_duplicates().shape[0])
print("Duplicate exact pairs:", processed.duplicated(["legal_norm", "simple_norm"]).sum())

raw_pairs = set(zip(raw["legal_norm"], raw["simple_norm"]))
processed_pairs = set(zip(processed["legal_norm"], processed["simple_norm"]))

print("\nRAW VS PROCESSED")
print("Processed pairs found in raw:", len(processed_pairs & raw_pairs))
print("Processed pairs missing from raw:", len(processed_pairs - raw_pairs))

variants = (
    raw.groupby("legal_norm")["simple_norm"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nORIGINAL TEXTS WITH MULTIPLE SIMPLIFICATIONS")
print("Number of originals with multiple versions:", (variants > 1).sum())

for original, count in variants[variants > 1].head(5).items():
    print(f"\nSimplification variants: {count}")
    print("Original:", original[:350])
    for simplified in raw.loc[
        raw["legal_norm"] == original, "simple_norm"
    ].drop_duplicates().head(3):
        print("Simplified:", simplified[:250])
