import json
from pathlib import Path

path = Path(
    "datasets/raw/SIMPLE-LAW-source/scripts/"
    "dataset_creation/Data_Generator.ipynb"
)

notebook = json.loads(path.read_text(encoding="utf-8"))

for i, cell in enumerate(notebook.get("cells", [])):
    source = "".join(cell.get("source", []))

    keywords = [
        "uk_351", "uk_600",
        "legal_text_IND_simplified",
        "legal_text_USA_simplified",
        "concat", "merge", "to_csv",
        "legal_text_UK_simplified"
    ]

    if any(keyword.lower() in source.lower() for keyword in keywords):
        print("\n" + "=" * 70)
        print(f"CELL {i} | TYPE: {cell.get('cell_type')}")
        print("=" * 70)
        print(source[:6000])

        for output in cell.get("outputs", []):
            text = output.get("text", [])
            if text:
                print("\nOUTPUT:")
                print("".join(text)[:2000])

print("\n" + "=" * 70)
print("GIT-TRACKED DATASET FILES")
print("=" * 70)
