#!/usr/bin/env python3
"""Leakage-resistant synthetic edit benchmark.

Groups every transformation of the same normalized source passage together,
uses a deterministic passage-level split, and excludes direct equality flags.
Still only evaluates controlled text edits, not legal meaning preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)


NEGATIONS = {"not", "no", "never", "neither", "nor"}
MODALS = {"shall", "must", "may", "might", "will", "should", "can", "cannot"}


def norm_source(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def bucket(source: str) -> str:
    digest = hashlib.sha1(norm_source(source).encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 100
    return "train" if value < 80 else ("validation" if value < 90 else "test")


def lexical_features(a: str, b: str) -> list[float]:
    ta = re.findall(r"\b\w+\b", a.lower())
    tb = re.findall(r"\b\w+\b", b.lower())
    sa, sb = set(ta), set(tb)
    union = sa | sb
    nums_a = re.findall(r"\b\d+(?:\.\d+)?\b", a)
    nums_b = re.findall(r"\b\d+(?:\.\d+)?\b", b)
    return [
        len(sa & sb) / max(len(union), 1),
        len(sa & sb) / max(len(sa), 1),
        min(len(ta), len(tb)) / max(len(ta), len(tb), 1),
        abs(len(ta) - len(tb)),
        abs(sum(t in NEGATIONS for t in ta) - sum(t in NEGATIONS for t in tb)),
        abs(sum(t in MODALS for t in ta) - sum(t in MODALS for t in tb)),
        float(nums_a != nums_b),
        # Deliberately no a == b feature.
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/experimental/lmp_controlled_synthetic.csv")
    ap.add_argument("--output", default="models/lmp_deadline_experimental/leakage_resistant")
    ap.add_argument("--max-train-rows", type=int, default=12000)
    ap.add_argument("--max-eval-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    needed = {"original_text", "simplified_text", "experimental_rating", "transformation_type"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    df["passage_group"] = df["original_text"].map(norm_source)
    df["passage_split"] = df["original_text"].map(bucket)

    # Confirm that each normalized source passage belongs to exactly one split.
    assert df.groupby("passage_group")["passage_split"].nunique().max() == 1

    train = df[df.passage_split.eq("train")].copy()
    val = df[df.passage_split.eq("validation")].copy()
    test = df[df.passage_split.eq("test")].copy()

    if len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    if len(val) > args.max_eval_rows:
        val = val.sample(args.max_eval_rows, random_state=args.seed)
    if len(test) > args.max_eval_rows:
        test = test.sample(args.max_eval_rows, random_state=args.seed)
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit("A passage-level split is empty; inspect data coverage.")

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                           max_features=40000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                           max_features=40000, sublinear_tf=True)
    word.fit(pd.concat([train.original_text, train.simplified_text]).astype(str))
    char.fit(pd.concat([train.original_text, train.simplified_text]).astype(str))

    def matrix(part):
        a = part.original_text.astype(str).tolist()
        b = part.simplified_text.astype(str).tolist()
        wa, wb = word.transform(a), word.transform(b)
        ca, cb = char.transform(a), char.transform(b)
        lf = csr_matrix(np.asarray(
            [lexical_features(x, y) for x, y in zip(a, b)], dtype=np.float32
        ))
        return hstack([
            wa, wb, abs(wa - wb), wa.multiply(wb),
            ca, cb, abs(ca - cb), ca.multiply(cb), lf,
        ], format="csr")

    print("Passage-group split (same original passage never crosses splits)")
    print(f"Train: {len(train)} rows / {train.passage_group.nunique()} passages")
    print(f"Validation: {len(val)} rows / {val.passage_group.nunique()} passages")
    print(f"Test: {len(test)} rows / {test.passage_group.nunique()} passages")

    x_train, x_val, x_test = matrix(train), matrix(val), matrix(test)

    reg = ExtraTreesRegressor(
        n_estimators=250, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=args.seed,
    )
    reg.fit(x_train, train.experimental_rating.astype(float))
    p_val = reg.predict(x_val)
    p_test = reg.predict(x_test)
    print("\nRegression:")
    print(f"Validation MAE: {mean_absolute_error(val.experimental_rating, p_val):.4f}")
    print(f"Test MAE: {mean_absolute_error(test.experimental_rating, p_test):.4f}")
    print(f"Test RMSE: {mean_squared_error(test.experimental_rating, p_test) ** 0.5:.4f}")

    y_train = train.transformation_type.eq("identity_control").astype(int)
    y_test = test.transformation_type.eq("identity_control").astype(int)
    clf = ExtraTreesClassifier(
        n_estimators=250, min_samples_leaf=2, max_features=0.8,
        class_weight="balanced", n_jobs=-1, random_state=args.seed,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs) if y_test.nunique() == 2 else None

    print("\nIdentity-vs-edited classification:")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"ROC-AUC: {auc if auc is not None else 'unavailable'}")
    print(classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["edited_control", "identity_control"],
        zero_division=0,
    ))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = test[[
        "pair_id", "source_document_id", "transformation_type",
        "experimental_rating", "passage_group",
    ]].copy()
    result["predicted_rating"] = p_test
    result["identity_probability"] = probs
    result.to_csv(out / "test_predictions.csv", index=False)
    metrics = {
        "test_rows": len(test),
        "test_passages": int(test.passage_group.nunique()),
        "regression_test_mae": float(mean_absolute_error(test.experimental_rating, p_test)),
        "regression_test_rmse": float(mean_squared_error(test.experimental_rating, p_test) ** 0.5),
        "binary_accuracy": float(accuracy_score(y_test, pred)),
        "binary_roc_auc": float(auc) if auc is not None else None,
        "direct_exact_equality_feature": False,
        "split_group": "normalized_original_passage",
        "warning": "Synthetic text-edit benchmark only; not legal meaning-preservation validation.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to: {out}")


if __name__ == "__main__":
    main()
