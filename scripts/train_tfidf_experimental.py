#!/usr/bin/env python3
"""Exploratory TF-IDF baseline using the XLM-R experimental split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.models.baselines import TfidfBaseline, save_tfidf
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

# Reproduce the exact seeded split used by train_lmp_experimental.py.
indices = np.random.default_rng(SEED).permutation(len(df))
n_train = int(len(df) * 0.70)
n_val = max(1, int(len(df) * 0.15))

train = df.iloc[indices[:n_train]].copy()
test = df.iloc[indices[n_train + n_val:]].copy()

print(f"Rows: {len(df)} | train: {len(train)} | test: {len(test)}")

baseline = TfidfBaseline(mapping="linear")
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

save_tfidf(
    baseline,
    OUT,
    {
        "n_train": int(len(train)),
        "languages": languages,
        "mapping": "linear",
        "fit_scope": "train_only",
        "description": (
            "Experimental TF-IDF word/character cosine baseline. "
            "Fitted on the training partition only; AI pseudo-labels."
        ),
    },
)

results = {
    "baseline": "TF-IDF word + character cosine",
    "seed": SEED,
    "split_strategy": "same_random_pair_split_as_experimental_XLMR",
    "n_total": int(len(df)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "label_provenance": "AI_generated_pseudo_labels_not_human_adjudicated",
    "document_leakage_possible": True,
    "test_metrics": metrics,
    "warning": (
        "Exploratory only: 30 pairs, AI pseudo-labels, "
        "random pair split and possible document leakage."
    ),
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tfidf_experimental_metrics.json").write_text(
    json.dumps(results, indent=2)
)

predictions = test[["pair_id", "source_document_id"]].copy()
predictions["actual_provisional_rating"] = actual
predictions["tfidf_predicted_rating"] = predicted
predictions.to_csv(OUT / "tfidf_test_predictions.csv", index=False)

print("\nTFIDF_TEST_METRICS")
print(json.dumps(metrics, indent=2))
print("\nBASELINE_SAVED:", OUT)
print("STATUS: SUCCESS")
