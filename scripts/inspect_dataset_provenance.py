import json
import re
from pathlib import Path
import pandas as pd

root = Path("datasets/raw/SIMPLE-LAW-source")

print("=" * 65)
print("RAW DATA FILE INSPECTION")
print("=" * 65)

for path in sorted((root / "data/raw").glob("*.csv")):
    try:
        df = pd.read_csv(path, nrows=5)
        print(f"\nFILE: {path.name}")
        print("Columns:", list(df.columns))
        print("Sample rows:", len(df))

        for col in df.columns:
            if col in df.columns:
                value = str(df.iloc[0][col])
                print(f"{col}: {value[:350].replace(chr(10), ' ')}")
    except Exception as e:
        print(f"Could not inspect {path.name}: {e}")

print("\n" + "=" * 65)
print("NOTEBOOK PROVENANCE REFERENCES")
print("=" * 65)

notebook_dir = root / "scripts/dataset_creation"

for path in sorted(notebook_dir.glob("*.ipynb")):
    print(f"\nNOTEBOOK: {path.name}")

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))

            if any(term in source.lower() for term in [
                "read_csv", "read_json", "github.com",
                "kaggle", "huggingface", "download",
                "public law", "uk law", "indian_laws"
            ]):
                lines = source.splitlines()
                for i, line in enumerate(lines):
                    if any(term in line.lower() for term in [
                        "read_csv", "read_json", "github.com",
                        "kaggle", "huggingface", "download",
                        "public law", "uk law", "indian_laws"
                    ]):
                        print("  " + line.strip()[:300])
    except Exception as e:
        print("Notebook inspection error:", e)
