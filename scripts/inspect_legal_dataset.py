import pandas as pd
from pathlib import Path
import hashlib

BASE = Path("datasets/raw/SIMPLE-LAW-source/data/processed")

def normalize(text):
    return " ".join(str(text).lower().split())

def fingerprint(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

train = pd.read_csv(BASE / "train_data.csv")
test = pd.read_csv(BASE / "test_data.csv")

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n--- {name} ---")
    print("Records:", len(df))
    print("Duplicate pairs:", df.duplicated(["legal_text", "simplified_text"]).sum())

train_originals = set(train.legal_text.map(fingerprint))
test_originals = set(test.legal_text.map(fingerprint))

train_pairs = set(zip(
    train.legal_text.map(fingerprint),
    train.simplified_text.map(fingerprint)
))
test_pairs = set(zip(
    test.legal_text.map(fingerprint),
    test.simplified_text.map(fingerprint)
))

print("\n--- LEAKAGE CHECK ---")
print("Original text overlap:", len(train_originals & test_originals))
print("Exact pair overlap:", len(train_pairs & test_pairs))

print("\n--- TEXT LENGTHS ---")
print("Training original text:")
print(train.legal_text.str.len().describe())

print("\nTesting original text:")
print(test.legal_text.str.len().describe())
