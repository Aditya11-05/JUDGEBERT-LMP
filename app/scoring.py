"""Score bounds and configurable preservation categories."""
from __future__ import annotations
import math

SCORE_MIN, SCORE_MAX = 1.0, 10.0


def bounded_score(z: float) -> float:
    """1 + 9*sigmoid(z): always inside [1, 10]. Numerically stable."""
    if z >= 0:
        s = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        s = e / (1.0 + e)
    return SCORE_MIN + (SCORE_MAX - SCORE_MIN) * s


def clip_score(x: float) -> float:
    return min(SCORE_MAX, max(SCORE_MIN, float(x)))


def categorize(score: float, preserved_min: float = 7.0, partial_min: float = 4.0) -> str:
    """Map a 1-10 score to a category. Thresholds are proposals, not a legal standard."""
    if not (SCORE_MIN <= score <= SCORE_MAX):
        raise ValueError(f"score {score} outside [{SCORE_MIN}, {SCORE_MAX}]")
    if partial_min >= preserved_min:
        raise ValueError("partial_min must be < preserved_min")
    if score >= preserved_min:
        return "Preserved"
    if score >= partial_min:
        return "Partially Preserved"
    return "Not Preserved"
