import pandas as pd
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

ranges = [
    (0, 15),
    (1300, 1340),
    (2100, 2140),
    (4100, 4140),
    (6250, 6285),
]

for start, end in ranges:
    print("\n" + "=" * 70)
    print(f"ROWS {start + 1} TO {end}")
    print("=" * 70)

    for i in range(start, min(end, len(df))):
        original = str(df.iloc[i]["legal_text"])
        simplified = str(df.iloc[i]["simplified_text"])

        print(f"\nROW {i + 1}")
        print("ORIGINAL:", original[:220].replace("\n", " "))
        print("SIMPLIFIED:", simplified[:160].replace("\n", " "))
