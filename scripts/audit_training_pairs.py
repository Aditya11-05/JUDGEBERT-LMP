import pandas as pd
from pathlib import Path

path = Path("datasets/raw/SIMPLE-LAW-source/data/raw/final_dataset.csv")
df = pd.read_csv(path).fillna("")

original = df["legal_text"].astype(str).str.strip()
simplified = df["simplified_text"].astype(str).str.strip()

valid = (original != "") & (simplified != "")

print("=" * 60)
print("TRAINING PAIR AUDIT")
print("=" * 60)

print("Total rows:", len(df))
print("Non-empty original texts:", (original != "").sum())
print("Non-empty simplified texts:", (simplified != "").sum())
print("Usable original-simplified pairs:", valid.sum())
print("Rows missing simplified text:", (~(simplified != "")).sum())
print("Rows missing original text:", (~(original != "")).sum())

pairs = pd.DataFrame({
    "original": original[valid],
    "simplified": simplified[valid]
})

print("\nExact duplicate pairs:", pairs.duplicated().sum())
print("Unique original texts:", pairs["original"].nunique())
print("Unique simplified texts:", pairs["simplified"].nunique())

print("\nPAIR COUNTS BY ROW RANGE")
for start, end in [
    (0, 1333),
    (1333, 2126),
    (2126, 4100),
    (4100, 6285)
]:
    subset = valid.iloc[start:end]
    print(
        f"Rows {start+1}-{end}: "
        f"{subset.sum()} usable pairs / {len(subset)} total"
    )

print("\nFIRST 10 ROWS WITH EMPTY SIMPLIFIED TEXT")
shown = 0
for i in range(len(df)):
    if simplified.iloc[i] == "":
        print(f"Row {i+1}: {original.iloc[i][:180]}")
        shown += 1
        if shown == 10:
            break
