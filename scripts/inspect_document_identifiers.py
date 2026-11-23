import pandas as pd
import re
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

patterns = {
    "UK Public Law": r"Public Law\s+\d+-\d+",
    "Indian Act": r"[A-Z][A-Za-z\s,()'-]+Act,?\s+\d{4}",
    "UK Act": r"[A-Z][A-Za-z\s,()'-]+Act\s+\d{4}",
    "Section heading": r"SECTION\s+\d+",
}

print("DOCUMENT IDENTIFIER INSPECTION")
print("=" * 55)

for name, pattern in patterns.items():
    matches = df["legal_text"].str.extractall(
        f"({pattern})", flags=re.IGNORECASE
    )[0].dropna()

    print(f"\n{name}: {len(matches)} matches")
    print("Examples:")
    for value in matches.drop_duplicates().head(10):
        print("-", value)

print("\nFIRST 15 ORIGINAL TEXTS")
for i, text in enumerate(df["legal_text"].head(15), 1):
    print(f"\n{i}. {str(text)[:350]}")
