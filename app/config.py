"""Runtime configuration. Everything can be overridden with JUDGEBERT_* environment variables (see .env.example)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .scoring import categorize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONE = "FacebookAI/xlm-roberta-base"


@dataclass(frozen=True)
class Settings:
    model_dir: Path = ROOT / "models" / "lmp"
    baseline_dir: Path = ROOT / "models" / "baselines"
    results_dir: Path = ROOT / "reports" / "results"
    config_path: Path = ROOT / "configs" / "default.yaml"
    backbone: str = DEFAULT_BACKBONE
    max_chars: int = 5000          # per text field
    max_batch: int = 100
    max_body_bytes: int = 2_000_000
    preserved_min: float | None = None   # None -> read from configs/default.yaml
    partial_min: float | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        kw: dict = {}
        for field, var in (("model_dir", "JUDGEBERT_MODEL_DIR"), ("baseline_dir", "JUDGEBERT_BASELINE_DIR"),
                           ("results_dir", "JUDGEBERT_RESULTS_DIR"), ("config_path", "JUDGEBERT_CONFIG")):
            if e.get(var):
                kw[field] = Path(e[var])
        if e.get("JUDGEBERT_BACKBONE"):
            kw["backbone"] = e["JUDGEBERT_BACKBONE"]
        for field, var in (("max_chars", "JUDGEBERT_MAX_CHARS"), ("max_batch", "JUDGEBERT_MAX_BATCH"),
                           ("max_body_bytes", "JUDGEBERT_MAX_BODY_BYTES")):
            if e.get(var):
                kw[field] = int(e[var])  # bad values fail at startup, not per request
        for field, var in (("preserved_min", "JUDGEBERT_PRESERVED_MIN"), ("partial_min", "JUDGEBERT_PARTIAL_MIN")):
            if e.get(var):
                kw[field] = float(e[var])
        return cls(**kw)

    def thresholds(self) -> tuple[float, float]:
        """(preserved_min, partial_min); validated so a bad config fails at startup."""
        pres, part = self.preserved_min, self.partial_min
        if pres is None or part is None:
            cats = (yaml.safe_load(self.config_path.read_text()) or {}).get("categories", {})
            pres = cats.get("preserved_min", 7.0) if pres is None else pres
            part = cats.get("partial_min", 4.0) if part is None else part
        categorize(5.0, pres, part)  # raises ValueError when partial_min >= preserved_min
        return float(pres), float(part)
