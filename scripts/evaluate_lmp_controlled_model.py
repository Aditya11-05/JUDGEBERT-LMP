#!/usr/bin/env python3
"""Evaluate the saved controlled-synthetic XLM-R model by edit category."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from app.models.xlmr import XLMRPairRegressor


class PairDataset(Dataset):
    def __init__(self, frame):
        self.left = frame["original_text"].astype(str).tolist()
        self.right = frame["simplified_text"].astype(str).tolist()
        self.y = frame["experimental_rating"].astype(float).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.left[idx], self.right[idx], self.y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/experimental/lmp_controlled_synthetic.csv")
    parser.add_argument("--model-dir", default="models/lmp_deadline_experimental")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest_path = model_dir / "lmp_manifest.json"
    weights_path = model_dir / "model_state.pt"
    tokenizer_dir = model_dir / "tokenizer"
    for p in (manifest_path, weights_path, tokenizer_dir):
        if not p.exists():
            raise SystemExit(f"Missing model artifact: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = pd.read_csv(args.data)
    data = data.loc[data["split"].eq("test")].copy()
    if data.empty:
        raise SystemExit("No rows found in the test split.")

    # Reproduce the deterministic test sample used by the training script.
    max_rows = int(manifest["rows_used"]["test"])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=int(manifest["seed"]))
    data = data.reset_index(drop=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = XLMRPairRegressor(
        backbone=manifest["backbone"],
        pretrained=False,
        config_path=manifest["backbone"],
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    loader = DataLoader(
        PairDataset(data), batch_size=args.batch_size, shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for left, right, _ in loader:
            encoded = tokenizer(
                list(left),
                list(right),
                padding=True,
                truncation=True,
                max_length=int(manifest["max_length"]),
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            predictions.extend(model(**encoded).detach().cpu().numpy().tolist())

    data["prediction"] = predictions
    data["absolute_error"] = (
        data["prediction"] - data["experimental_rating"]
    ).abs()
    data["control_group"] = np.where(
        data["transformation_type"].eq("identity_control"),
        "identity_control",
        "changed_text_control",
    )

    print("\n=== Results by transformation ===")
    by_type = data.groupby("transformation_type").agg(
        n=("prediction", "size"),
        target_mean=("experimental_rating", "mean"),
        prediction_mean=("prediction", "mean"),
        prediction_std=("prediction", "std"),
        mae=("absolute_error", "mean"),
    ).round(3)
    print(by_type.to_string())

    print("\n=== Identity vs changed-text controls ===")
    by_group = data.groupby("control_group").agg(
        n=("prediction", "size"),
        target_mean=("experimental_rating", "mean"),
        prediction_mean=("prediction", "mean"),
        prediction_std=("prediction", "std"),
        mae=("absolute_error", "mean"),
    ).round(3)
    print(by_group.to_string())

    identity_mean = data.loc[
        data["control_group"].eq("identity_control"), "prediction"
    ].mean()
    changed_mean = data.loc[
        data["control_group"].eq("changed_text_control"), "prediction"
    ].mean()
    print("\n=== Separation check ===")
    print(f"Mean identity prediction:      {identity_mean:.3f}")
    print(f"Mean changed-text prediction:  {changed_mean:.3f}")
    print(f"Identity minus changed:        {identity_mean - changed_mean:.3f}")

    output_path = model_dir / "controlled_test_predictions.csv"
    data.to_csv(output_path, index=False)
    print(f"\nSaved row-level predictions: {output_path}")


if __name__ == "__main__":
    main()
