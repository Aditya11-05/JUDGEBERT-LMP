import pandas as pd
import re
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/data")

pairs = pd.read_csv(base / "raw/final_dataset.csv").fillna("")
pairs = pairs[
    (pairs["legal_text"].astype(str).str.strip() != "") &
    (pairs["simplified_text"].astype(str).str.strip() != "")
].copy()

source = pd.read_csv(base / "raw/indian_laws.csv").fillna("")

def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

documents = []

for _, row in source.iterrows():
    text = norm(row["legal_text"])
    if len(text) > 100:
        title = str(row["legal_text"]).splitlines()[0].strip()
        documents.append((title, text))

matched = []
unmatched = []

for index, row in pairs.iterrows():
    original = norm(row["legal_text"])
    prefix = original[:120]

    if len(prefix) < 120:
        unmatched.append(index + 1)
        continue

    found = None
    for title, document in documents:
        if prefix in document:
            found = title
            break

    if found:
        matched.append((index + 1, found))
    else:
        unmatched.append(index + 1)

print("TOTAL PAIRED RECORDS:", len(pairs))
print("MATCHED TO INDIAN ACTS:", len(matched))
print("NOT MATCHED:", len(unmatched))

print("\nMATCHED ROWS:")
for row_number, title in matched[:100]:
    print(f"Row {row_number}: {title}")

print("\nUNMATCHED ROW NUMBERS:")
print(unmatched[:200])

print("\nUNMATCHED SAMPLE TEXT:")
for row_number in unmatched[:20]:
    text = pairs.loc[row_number - 1, "legal_text"]
    print(f"\nRow {row_number}: {str(text)[:250].replace(chr(10), ' ')}")
