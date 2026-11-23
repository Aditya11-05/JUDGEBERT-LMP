#!/usr/bin/env python3
"""Experimental multilingual sentence-embedding baseline using the XLM-R split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.models.baselines import (
    MultilingualEmbeddingBaseline,
    save_multilingual,
)
from app.models.evaluate import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/JUDGEBERT_LMP_30_pairs_AI_reviewed_provisional.csv"
OUT = ROOT / "models/baselines"
SEED = 42

df = pd.read_csv(DATA)
df["adjudicated_rating"] = pd.to_numeric(
    df["adjudicated_rating"], errors="coerce"
)

if df["adjudicated_rating"].isna().any():
    raise SystemExit("Missing or invalid provisional ratings.")

indices = np.random.default_rng(SEED).permutation(len(df))
n_train = int(len(df) * 0.70)
n_val = max(1, int(len(df) * 0.15))

train = df.iloc[indices[:n_train]].copy()
test = df.iloc[n_train + n_val:].copy()

print(f"Rows: {len(df)} | train: {len(train)} | test: {len(test)}")
print("Loading multilingual sentence encoder...")

baseline = MultilingualEmbeddingBaseline()
baseline.fit(
    train["original_text"].astype(str).tolist(),
    train["simplified_text"].astype(str).tolist(),
    train["adjudicated_rating"].to_numpy(dtype=float),
)

originals = test["original_text"].astype(str).tolist()
simplified = test["simplified_text"].astype(str).tolist()
actual = test["adjudicated_rating"].to_numpy(dtype=float)
predicted = baseline.predict(originals, simplified)
metrics = regression_metrics(actual, predicted)

languages = sorted(train["language"].dropna().unique().tolist())

save_multilingual(
    baseline,
    OUT,
    {
        "n_train": int(len(train)),
        "languages": languages,
        "fit_scope": "train_only",
        "model_name": baseline.model_name,
        "description": (
            "Experimental multilingual MiniLM cosine-similarity baseline. "
            "Linear score mapping fitted on training partition only; "
            "AI pseudo-labels."
        ),
    },
)

results = {
    "baseline": "Multilingual MiniLM sentence-embedding cosine",
    "model_name": baseline.model_name,
    "seed": SEED,
    "split_strategy": "same_random_pair_split_as_experimental_XLMR_and_TFIDF",
    "n_total": int(len(df)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "languages": languages,
    "label_provenance": "AI_generated_pseudo_labels_not_human_adjudicated",
    "document_leakage_possible": True,
    "test_metrics": metrics,
    "warning": (
        "Exploratory only: 30 pairs, AI pseudo-labels, "
        "random pair split and possible document leakage."
    ),
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "multilingual_embedding_experimental_metrics.json").write_text(
    json.dumps(results, indent=2)
)

predictions = test[["pair_id", "source_document_id"]].copy()
predictions["actual_provisional_rating"] = actual
predictions["multilingual_embedding_predicted_rating"] = predicted
predictions.to_csv(
    OUT / "multilingual_embedding_test_predictions.csv",
    index=False,
)

print("\nMULTILINGUAL_EMBEDDING_TEST_METRICS")
print(json.dumps(metrics, indent=2))
print("\nBASELINE_SAVED:", OUT)
print("STATUS: SUCCESS")
