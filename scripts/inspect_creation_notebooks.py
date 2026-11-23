import json
from pathlib import Path

base = Path("datasets/raw/SIMPLE-LAW-source/scripts/dataset_creation")

files = [
    "dataset_generation.ipynb",
    "explore_merge_datasets.ipynb",
    "split_dataset.ipynb",
]

for filename in files:
    path = base / filename
    print("\n" + "=" * 80)
    print("NOTEBOOK:", filename)

    notebook = json.loads(path.read_text(encoding="utf-8"))

    for i, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        if source.strip():
            print(f"\n--- CELL {i} ({cell['cell_type']}) ---")
            print(source[:2500])
