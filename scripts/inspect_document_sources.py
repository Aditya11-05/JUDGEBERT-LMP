import pandas as pd
import re
from pathlib import Path
from collections import Counter

base = Path("datasets/raw/SIMPLE-LAW-source/data/processed")

files = [
    "INDIAN_LAW_DATASET.csv",
    "simplification_final_dataset.csv"
]

for filename in files:
    df = pd.read_csv(base / filename)

    print(f"\n{'=' * 70}")
    print(f"SOURCE: {filename}")
    print(f"TOTAL ROWS: {len(df)}")

    references = Counter()

    for text in df["legal_text"].dropna():
        text = str(text)

        # Look for Act titles and common legal document references
        matches = re.findall(
            r"(?i)(?:the\s+)?[A-Z][A-Za-z0-9(),&'’ -]{2,100}?\bAct(?:,?\s+\d{4})?",
            text[:1500]
        )

        for match in matches:
            cleaned = " ".join(match.split())
            references[cleaned] += 1

    print("\nMost frequent possible document references:")
    for reference, count in references.most_common(15):
        print(f"{count:5} | {reference}")

