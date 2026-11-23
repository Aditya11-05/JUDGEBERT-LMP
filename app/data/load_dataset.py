from __future__ import annotations
import pandas as pd
from .validate_dataset import validate


def load_dataset(path: str, require_ratings: bool = False) -> pd.DataFrame:
    df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_csv(path)
    problems = validate(df, require_ratings=require_ratings)
    if problems:
        raise ValueError("invalid dataset: " + "; ".join(problems))
    return df
