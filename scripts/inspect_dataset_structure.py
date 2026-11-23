import pandas as pd
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data/processed")

for filename in [
    "INDIAN_LAW_DATASET.csv",
    "simplification_final_dataset.csv",
    "merged_dataset.csv"
]:
    df = pd.read_csv(base / filename)

    print(f"\n{'=' * 60}")
    print(filename)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nFirst 3 original texts:")
    for i, text in enumerate(df["legal_text"].head(3), 1):
        print(f"\n{i}. {str(text)[:500]}")

