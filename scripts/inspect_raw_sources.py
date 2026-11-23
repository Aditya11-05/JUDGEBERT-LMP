import pandas as pd
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data/raw")

for path in sorted(base.glob("*.csv")):
    print("\n" + "=" * 75)
    print("FILE:", path.name)

    try:
        df = pd.read_csv(path, nrows=5, low_memory=False)

        print("Columns:", df.columns.tolist())
        print("Sample shape:", df.shape)

        for i, row in df.head(2).iterrows():
            print(f"\nRecord {i + 1}:")
            for col, value in row.items():
                print(f"{col}: {str(value)[:250]}")

    except Exception as e:
        print("Could not inspect:", e)
