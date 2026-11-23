import pandas as pd
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data/processed")

for filename in [
    "INDIAN_LAW_DATASET.csv",
    "simplification_final_dataset.csv"
]:
    df = pd.read_csv(base / filename)

    print("\n" + "=" * 80)
    print("FILE:", filename)

    for i, text in enumerate(df["legal_text"].head(12), 1):
        lines = str(text).splitlines()
        print(f"\n--- RECORD {i} ---")
        for line in lines[:5]:
            print(line[:250])

