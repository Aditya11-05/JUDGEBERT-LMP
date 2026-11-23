#!/usr/bin/env python3
"""Train an isolated experimental XLM-R LMP regressor on controlled pairs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from app.models.xlmr import XLMRPairRegressor


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.original = frame["original_text"].astype(str).tolist()
        self.changed = frame["simplified_text"].astype(str).tolist()
        self.labels = frame["experimental_rating"].astype(float).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.original[index], self.changed[index], self.labels[index]


def safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    value = fn(y_true, y_pred).statistic
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/experimental/lmp_controlled_synthetic.csv")
    parser.add_argument("--output", default="models/lmp_deadline_experimental")
    parser.add_argument("--backbone", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int, default=8000)
    parser.add_argument("--max-eval-rows", type=int, default=1000)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    frame = pd.read_csv(data_path)
    required = {
        "original_text", "simplified_text", "experimental_rating",
        "source_document_id", "split", "label_provenance",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing dataset columns: {sorted(missing)}")
    if not frame["label_provenance"].eq("controlled_synthetic_transformation").all():
        raise SystemExit("Refusing mixed or non-synthetic label provenance.")
    if not frame["experimental_rating"].between(1, 10).all():
        raise SystemExit("Ratings must be in the 1–10 range.")

    # Keep the generator's document-level split intact.
    train = frame.loc[frame["split"] == "train"].copy()
    valid = frame.loc[frame["split"] == "validation"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise SystemExit("Train, validation, and test splits must all be nonempty.")

    if len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    if len(valid) > args.max_eval_rows:
        valid = valid.sample(args.max_eval_rows, random_state=args.seed)
    if len(test) > args.max_eval_rows:
        test = test.sample(args.max_eval_rows, random_state=args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Rows used — train: {len(train)}, validation: {len(valid)}, test: {len(test)}")
    print("Loading tokenizer and pretrained XLM-R backbone...")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = XLMRPairRegressor(backbone=args.backbone, pretrained=True).to(device)

    def collate(batch):
        left = [item[0] for item in batch]
        right = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        encoded = tokenizer(
            left, right, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        return encoded, labels

    train_loader = DataLoader(
        PairDataset(train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
    )
    valid_loader = DataLoader(
        PairDataset(valid), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
    )
    test_loader = DataLoader(
        PairDataset(test), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
    )

    # Memory-safe baseline: freeze the pretrained encoder and train only the
    # regression head. This is not full XLM-R fine-tuning.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    head_params = list(model.regressor.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()

    def evaluate(loader):
        model.eval()
        actual, predicted = [], []
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for encoded, labels in loader:
                encoded = {k: v.to(device) for k, v in encoded.items()}
                labels = labels.to(device)
                outputs = model(**encoded)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * len(labels)
                total += len(labels)
                actual.extend(labels.cpu().numpy().tolist())
                predicted.extend(outputs.cpu().numpy().tolist())
        return np.asarray(actual), np.asarray(predicted), total_loss / max(total, 1)

    best_val = float("inf")
    best_state = None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()  # Frozen encoder: disable its dropout during head training.
        running, count = 0.0, 0
        for step, (encoded, labels) in enumerate(train_loader, start=1):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**encoded)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(labels)
            count += len(labels)
            if step % 100 == 0:
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                      f"train_loss={running / count:.4f}")

        y_val, p_val, val_loss = evaluate(valid_loader)
        train_loss = running / max(count, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"validation_loss={val_loss:.4f}, "
              f"validation_MAE={mean_absolute_error(y_val, p_val):.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test, test_loss = evaluate(test_loader)
    metrics = {
        "n": int(len(y_test)),
        "mse": float(test_loss),
        "rmse": float(np.sqrt(mean_squared_error(y_test, p_test))),
        "mae": float(mean_absolute_error(y_test, p_test)),
        "pearson": safe_corr(pearsonr, y_test, p_test),
        "spearman": safe_corr(spearmanr, y_test, p_test),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model_state.pt")
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {
        "version": "xlmr-lmp-controlled-experimental-frozen-encoder-0.1",
        "experimental": True,
        "encoder_frozen": True,
        "backbone": args.backbone,
        "weights_file": "model_state.pt",
        "tokenizer_dir": "tokenizer",
        "max_length": args.max_length,
        "score_min": 1,
        "score_max": 10,
        "rating_column": "experimental_rating",
        "label_provenance": "controlled_synthetic_transformation_not_human_adjudicated",
        "adjudication_status": "not_human_adjudicated",
        "split_strategy": "precomputed_source_document_group_hash",
        "document_leakage_possible": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "rows_used": {
            "train": len(train), "validation": len(valid), "test": len(test)
        },
        "source_documents": int(frame["source_document_id"].nunique()),
        "test_metrics": metrics,
        "training_seconds": round(time.time() - started, 1),
        "warning": (
            "Frozen-encoder baseline: only the regression head was trained. "
            "Controlled synthetic benchmark only. Scores reflect the supplied "
            "text-edit rules, not validated legal meaning preservation or "
            "generalisation to real-world simplifications."
        ),
    }
    (output / "lmp_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nSaved experimental model to:", output)
    print("Test metrics:", json.dumps(metrics, indent=2))
    print("Existing models/lmp checkpoint was not modified.")


if __name__ == "__main__":
    main()
