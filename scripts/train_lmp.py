#!/usr/bin/env python3
"""Train XLM-R on adjudicated human LMP ratings; never fabricates labels."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.validate_dataset import validate, validate_training_data
from app.data.split_dataset import group_split, assert_no_leakage
from app.models.xlmr import XLMRPairRegressor
from app.models.evaluate import regression_metrics

class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        enc = self.tokenizer(str(r.original_text), str(r.simplified_text), truncation=True,
                             max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(float(r.adjudicated_rating), dtype=torch.float32)

def collate(batch):
    keys = batch[0][0].keys()
    return {k: torch.stack([x[0][k] for x in batch]) for k in keys}, torch.stack([x[1] for x in batch])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with adjudicated_rating and document IDs")
    ap.add_argument("--output", default=str(ROOT / "models" / "lmp"))
    ap.add_argument("--backbone", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df = pd.read_csv(args.data)
    problems = validate(df, require_ratings=False)
    if problems:
        raise SystemExit("Dataset validation failed: " + " | ".join(problems))

    training_problems = validate_training_data(df)
    if training_problems:
        raise SystemExit(
            "Strict training-data validation failed: "
            + " | ".join(training_problems)
        )

    if len(df) < 30:
        raise SystemExit(
            f"Only {len(df)} eligible adjudicated pairs. "
            "Collect more human annotations before training."
        )
    df = group_split(df, seed=args.seed)
    assert_no_leakage(df)
    train, val, test = (df[df.split == s].copy() for s in ("train", "val", "test"))
    if min(len(train), len(val), len(test)) == 0: raise SystemExit("Group split produced an empty partition; add more source documents.")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = XLMRPairRegressor(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    train_dl = DataLoader(PairDataset(train, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(PairDataset(val, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for x, y in train_dl:
            x={k:v.to(device) for k,v in x.items()}; y=y.to(device)
            opt.zero_grad(); pred=model(**x); loss=loss_fn(pred, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(loss.item())
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in val_dl:
                x={k:v.to(device) for k,v in x.items()}; ps.extend(model(**x).cpu().numpy()); ys.extend(y.numpy())
        metrics=regression_metrics(ys, ps)
        print(f"epoch={epoch+1} train_mse={np.mean(losses):.4f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f}")
        if metrics["rmse"] < best:
            best=metrics["rmse"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    test_dl=DataLoader(PairDataset(test, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in test_dl:
            x={k:v.to(device) for k,v in x.items()}; ps.extend(model(**x).cpu().numpy()); ys.extend(y.numpy())
    test_metrics=regression_metrics(ys,ps)
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model_state.pt")
    tokenizer.save_pretrained(out / "tokenizer")
    model.encoder.config.save_pretrained(out / "tokenizer")
    manifest={"version":"xlmr-lmp-1.0.0","backbone":args.backbone,"weights_file":"model_state.pt",
              "tokenizer_dir":"tokenizer","max_length":args.max_length,"score_min":1,"score_max":10,
              "rating_column":"adjudicated_rating","label_provenance":"human_adjudicated",
              "split_seed":args.seed,"n_train":len(train),"n_val":len(val),"n_test":len(test),
              "document_counts":{"train":int(train.source_document_id.nunique()),"val":int(val.source_document_id.nunique()),"test":int(test.source_document_id.nunique())},
              "test_metrics":test_metrics}
    (out / "lmp_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("TEST_METRICS",json.dumps(test_metrics)); print("MODEL_SAVED",out)
if __name__ == "__main__": main()
