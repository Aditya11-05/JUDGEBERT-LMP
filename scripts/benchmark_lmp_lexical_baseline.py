#!/usr/bin/env python3
"""Benchmark transparent lexical baselines on the controlled synthetic LMP set.

This tests detection of the generator's controlled edits only. It is not a
validated legal-meaning-preservation model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from scipy.sparse import hstack, csr_matrix


NEGATIONS = {"not", "no", "never", "neither", "nor"}
MODALS = {"shall", "must", "may", "might", "will", "should", "can", "cannot"}


def lexical_features(a: str, b: str) -> list[float]:
    ta = re.findall(r"\b\w+\b", a.lower())
    tb = re.findall(r"\b\w+\b", b.lower())
    sa, sb = set(ta), set(tb)
    union = sa | sb
    jaccard = len(sa & sb) / max(len(union), 1)
    overlap = len(sa & sb) / max(len(sa), 1)
    length_ratio = min(len(ta), len(tb)) / max(len(ta), len(tb), 1)
    neg_a = sum(t in NEGATIONS for t in ta)
    neg_b = sum(t in NEGATIONS for t in tb)
    modal_a = sum(t in MODALS for t in ta)
    modal_b = sum(t in MODALS for t in tb)
    nums_a = re.findall(r"\b\d+(?:\.\d+)?\b", a)
    nums_b = re.findall(r"\b\d+(?:\.\d+)?\b", b)
    return [
        jaccard, overlap, length_ratio,
        abs(len(ta) - len(tb)),
        abs(neg_a - neg_b),
        abs(modal_a - modal_b),
        float(nums_a != nums_b),
        float(a.strip() == b.strip()),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/experimental/lmp_controlled_synthetic.csv")
    parser.add_argument("--output", default="models/lmp_deadline_experimental/lexical_baseline")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-eval-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = {"original_text", "simplified_text", "experimental_rating", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    train = df[df.split.eq("train")].copy()
    val = df[df.split.eq("validation")].copy()
    test = df[df.split.eq("test")].copy()
    if len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    if len(val) > args.max_eval_rows:
        val = val.sample(args.max_eval_rows, random_state=args.seed)
    if len(test) > args.max_eval_rows:
        test = test.sample(args.max_eval_rows, random_state=args.seed)

    # Fit vectorizers on training texts only; evaluate on held-out documents.
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                           sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                           max_features=50000, sublinear_tf=True)

    train_text = pd.concat([train.original_text, train.simplified_text]).astype(str)
    word.fit(train_text)
    char.fit(train_text)

    def make_matrix(part):
        a = part.original_text.astype(str).tolist()
        b = part.simplified_text.astype(str).tolist()
        wa, wb = word.transform(a), word.transform(b)
        ca, cb = char.transform(a), char.transform(b)
        # Sparse pair representation: both sides, absolute difference,
        # elementwise product, plus explicit scalar lexical indicators.
        lexical = np.asarray([lexical_features(x, y) for x, y in zip(a, b)],
                             dtype=np.float32)
        lexical_sparse = csr_matrix(lexical)
        return hstack([
            wa, wb, abs(wa - wb), wa.multiply(wb),
            ca, cb, abs(ca - cb), ca.multiply(cb),
            lexical_sparse,
        ], format="csr")

    print(f"Rows — train {len(train)}, validation {len(val)}, test {len(test)}")
    x_train = make_matrix(train)
    x_val = make_matrix(val)
    x_test = make_matrix(test)

    # Main benchmark: regress on the known synthetic 1–10 targets.
    reg = ExtraTreesRegressor(
        n_estimators=250, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=args.seed,
    )
    reg.fit(x_train, train.experimental_rating.astype(float))
    p_val = reg.predict(x_val)
    p_test = reg.predict(x_test)
    print("\nRegression benchmark:")
    print(f"Validation MAE: {mean_absolute_error(val.experimental_rating, p_val):.4f}")
    print(f"Test MAE:       {mean_absolute_error(test.experimental_rating, p_test):.4f}")
    print(f"Test RMSE:      {mean_squared_error(test.experimental_rating, p_test) ** 0.5:.4f}")

    # Binary task directly tests the intended controlled distinction.
    y_train = train.transformation_type.eq("identity_control").astype(int)
    y_test = test.transformation_type.eq("identity_control").astype(int)
    clf = ExtraTreesClassifier(
        n_estimators=250, min_samples_leaf=2, max_features=0.8,
        class_weight="balanced", n_jobs=-1, random_state=args.seed,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    print("\nBinary identity-vs-edited benchmark:")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    try:
        print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.4f}")
    except ValueError:
        print("ROC-AUC: unavailable (only one class in test sample)")
    print(classification_report(
        y_test, pred, target_names=["edited_control", "identity_control"],
        zero_division=0,
    ))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = test[[
        "pair_id", "source_document_id", "transformation_type",
        "experimental_rating",
    ]].copy()
    results["predicted_rating"] = p_test
    results["identity_probability"] = probs
    results.to_csv(output / "test_predictions.csv", index=False)
    summary = {
        "experimental_only": True,
        "test_rows": len(test),
        "regression_test_mae": float(mean_absolute_error(
            test.experimental_rating, p_test)),
        "regression_test_rmse": float(
            mean_squared_error(test.experimental_rating, p_test) ** 0.5),
        "binary_accuracy": float(accuracy_score(y_test, pred)),
        "binary_roc_auc": (
            float(roc_auc_score(y_test, probs))
            if len(np.unique(y_test)) == 2 else None
        ),
        "warning": (
            "This benchmark detects synthetic text-edit patterns only. "
            "It does not validate legal meaning preservation."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved predictions and metrics under: {output}")


if __name__ == "__main__":
    main()
