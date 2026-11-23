import pandas as pd
import re
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

patterns = [
    r"Public Law\s+\d+-\d+",
    r"An Act To",
    r"An Act Entitled",
    r"THE\s+[A-Z][A-Z\s,()'-]{5,}\s+ACT",
]

print("POTENTIAL DOCUMENT HEADERS")
print("=" * 60)

found = 0

for i, text in enumerate(df["legal_text"]):
    text = str(text)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"\nRow: {i + 1}")
            print("Matched:", match.group(0))
            print("Text:", text[:500].replace("\n", " "))
            found += 1
            break

    if found >= 20:
        break

print(f"\nTotal displayed: {found}")
