import pandas as pd
import re
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

pattern = re.compile(r"Public Law\s+\d+-\d+", re.IGNORECASE)

matches = 0

print("PUBLIC LAW IDENTIFIER LOCATIONS")
print("=" * 60)

for i, text in enumerate(df["legal_text"]):
    text = str(text)
    found = pattern.search(text)

    if found:
        matches += 1
        if matches <= 20:
            print(f"\nRow: {i + 1}")
            print("Identifier:", found.group(0))
            print("At beginning:", found.start() < 80)
            print("Text:", text[:450].replace("\n", " "))

print("\nTotal rows containing Public Law:", matches)

print("\nDATASET ROW RANGE CHECK")
for start, end in [(0, 100), (2000, 2200), (4000, 4200), (6000, 6285)]:
    subset = df.iloc[start:end]
    count = subset["legal_text"].str.contains(
        pattern, na=False
    ).sum()
    print(f"Rows {start+1}-{end}: {count} Public Law references")
