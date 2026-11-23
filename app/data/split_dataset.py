"""Reproducible group-aware splitting: a source document never spans two partitions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def group_split(df: pd.DataFrame, seed: int = 42, train: float = 0.70, val: float = 0.15,
                group_col: str = "source_document_id") -> pd.DataFrame:
    if not (0 < train and 0 < val and train + val < 1.0):
        raise ValueError("train + val must be < 1")
    groups = sorted(df[group_col].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    sizes = df.groupby(group_col).size().to_dict()
    total = len(df)
    assign, count = {}, 0
    for g in groups:  # fill train, then val, remainder -> test
        if count < train * total:
            assign[g] = "train"
        elif count < (train + val) * total:
            assign[g] = "val"
        else:
            assign[g] = "test"
        count += sizes[g]
    out = df.copy()
    out["split"] = out[group_col].map(assign)
    return out


def assert_no_leakage(df: pd.DataFrame, group_col: str = "source_document_id") -> None:
    per = df.groupby(group_col)["split"].nunique()
    leaked = per[per > 1]
    if len(leaked):
        raise AssertionError(f"documents in more than one split: {list(leaked.index)[:5]}")
