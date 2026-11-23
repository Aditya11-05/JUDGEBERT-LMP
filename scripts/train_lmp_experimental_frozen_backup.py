#!/usr/bin/env python3
"""Experimental XLM-R training with explicitly disclosed AI pseudo-labels."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models.xlmr import XLMRPairRegressor
from app.models.evaluate import regression_metrics


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        encoded = self.tokenizer(
            str(row.original_text),
            str(row.simplified_text),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in encoded.items()}
        target = torch.tensor(
            float(row.adjudicated_rating), dtype=torch.float32
        )
        return inputs, target


def collate(batch):
    keys = batch[0][0].keys()
    x = {k: torch.stack([item[0][k] for item in batch]) for k in keys}
    y = torch.stack([item[1] for item in batch])
    return x, y


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            predicted.extend(model(**x).cpu().numpy().tolist())
            actual.extend(y.numpy().tolist())
    return actual, predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=str(ROOT / "models" / "lmp"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--backbone", default="FacebookAI/xlm-roberta-base")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)

    required = [
        "pair_id", "source_document_id", "language",
        "original_text", "simplified_text", "adjudicated_rating"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    if df["pair_id"].duplicated().any():
        raise SystemExit("Duplicate pair_id values.")

    for col in ("original_text", "simplified_text"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise SystemExit(f"Empty values in {col}.")

    if not df["language"].isin({"en", "hi"}).all():
        raise SystemExit("Unsupported language values.")


    if "adjudicated_rating" not in df.columns:
        raise SystemExit("Missing adjudicated_rating column.")

    df["adjudicated_rating"] = pd.to_numeric(
        df["adjudicated_rating"], errors="coerce"
    )
    if df["adjudicated_rating"].isna().any():
        raise SystemExit("Some provisional ratings are missing or invalid.")

    if len(df) < 10:
        raise SystemExit("At least 10 pairs are needed for this experiment.")

    # Deliberately random pair split: document-level split is impossible
    # with only two source documents. This limitation is recorded below.
    indices = np.random.default_rng(args.seed).permutation(len(df))
    n_train = int(len(df) * 0.70)
    n_val = max(1, int(len(df) * 0.15))

    train = df.iloc[indices[:n_train]].copy()
    val = df.iloc[indices[n_train:n_train + n_val]].copy()
    test = df.iloc[indices[n_train + n_val:]].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit("Random split produced an empty partition.")

    print(
        f"Rows: {len(df)} | train: {len(train)} | "
        f"validation: {len(val)} | test: {len(test)}"
    )
    print("WARNING: AI pseudo-labels; random pair split; document leakage possible.")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    model = XLMRPairRegressor(args.backbone).to(device)

    # Freeze pretrained encoder; train only the regression head.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.regressor.parameters(), lr=args.learning_rate
    )
    print("Training mode: frozen XLM-R encoder + trainable regression head")
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(
        PairDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        PairDataset(val, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()
        losses = []

        for x, y in train_loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            optimizer.zero_grad()
            predictions = model(**x)
            loss = loss_fn(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        actual, predicted = predict(model, val_loader, device)
        metrics = regression_metrics(actual, predicted)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_mse={np.mean(losses):.4f} | "
            f"val_mae={metrics['mae']:.4f} | "
            f"val_rmse={metrics['rmse']:.4f}"
        )

        if metrics["rmse"] < best_loss:
            best_loss = metrics["rmse"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise SystemExit("Training did not produce a usable model.")

    model.load_state_dict(best_state)
    test_loader = DataLoader(
        PairDataset(test, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    actual, predicted = predict(model, test_loader, device)
    test_metrics = regression_metrics(actual, predicted)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, output / "model_state.pt")
    tokenizer.save_pretrained(output / "tokenizer")
    model.encoder.config.save_pretrained(output / "tokenizer")

    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "version": "xlmr-lmp-experimental-0.1",
        "experimental": True,
        "backbone": args.backbone,
        "weights_file": "model_state.pt",
        "tokenizer_dir": "tokenizer",
        "max_length": args.max_length,
        "score_min": 1,
        "score_max": 10,
        "rating_column": "adjudicated_rating",
        "label_provenance": "AI_generated_pseudo_labels_not_human_adjudicated",
        "adjudication_status": "not_human_adjudicated",
        "split_strategy": "random_pair_split",
        "document_leakage_possible": True,
        "source_data_sha256": source_hash,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_total": len(df),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "source_documents": int(df.source_document_id.nunique()),
        "test_metrics": test_metrics,
        "metrics_warning": (
            "Exploratory only. Small AI-pseudo-labelled dataset; "
            "random split; not evidence of legal-domain generalisation."
        ),
    }
    (output / "lmp_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nEXPERIMENTAL_TEST_METRICS", json.dumps(test_metrics))
    print("MODEL_SAVED", output.resolve())


if __name__ == "__main__":
    main()
