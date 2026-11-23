"""Metrics: Pearson, Spearman, RMSE, MAE (+ optional category accuracy / macro-F1)."""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y) != len(p) or len(y) < 3:
        raise ValueError("need >= 3 aligned samples")
    const = np.std(y) == 0 or np.std(p) == 0
    return {
        "n": int(len(y)),
        "pearson": None if const else float(pearsonr(y, p)[0]),
        "spearman": None if const else float(spearmanr(y, p)[0]),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
    }
