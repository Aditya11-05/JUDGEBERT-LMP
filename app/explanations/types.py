"""Data types for the rule-based legal-error explanation module.

Nothing here depends on a language or on the LMP regression model.
"""
from __future__ import annotations
from dataclasses import dataclass, field

STRENGTHS = ("strong", "moderate", "weak")  # categorical evidence strength, NOT a probability

ERROR_TYPES = (
    "obligation_permission_change",
    "negation_change",
    "date_or_deadline_change",
    "bound_or_direction_change",
    "monetary_amount_change",
    "percentage_change",
    "numerical_value_change",
    "frequency_change",
    "party_or_entity_change",
    "scope_change",
    "condition_removed",
    "condition_added",
    "condition_strength_change",
    "exception_removed",
    "exception_added",
    "possible_omission_or_addition",
)
# Extra labels human annotators may use that no rule detects (kept for the dataset schema).
HUMAN_ONLY_TYPES = ("contradiction", "hallucination", "other", "none")

DISCLAIMER = (
    "Rule-based hints from surface patterns. They are not legal advice, do not establish a legal "
    "conclusion, and are not an explanation of the LMP regression model's internal reasoning."
)


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class Mention:
    """One extracted item (a modal verb, an amount, a date, ...) with its position."""
    kind: str
    key: object
    span: Span
    attrs: dict = field(default_factory=dict)


@dataclass
class Features:
    text: str
    mentions: dict = field(default_factory=dict)
    neg_consumed: list = field(default_factory=list)  # (start, end) ranges already explained by a modal/exception
    content_tokens: int = 0
    passive: bool = False

    def add(self, m: Mention) -> None:
        self.mentions.setdefault(m.kind, []).append(m)

    def get(self, kind: str) -> list:
        return sorted(self.mentions.get(kind, []), key=lambda m: m.span.start)


@dataclass
class Finding:
    error_type: str
    change_kind: str  # "changed" | "omitted" | "added"
    original_span: Span | None
    simplified_span: Span | None
    explanation: str
    evidence_strength: str  # see STRENGTHS
    rule_id: str
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "change_kind": self.change_kind,
            "original_span": self.original_span.to_dict() if self.original_span else None,
            "simplified_span": self.simplified_span.to_dict() if self.simplified_span else None,
            "explanation": self.explanation,
            "evidence_strength": self.evidence_strength,
            "rule_id": self.rule_id,
            "language": self.language,
        }


@dataclass
class ExplanationResult:
    language: str
    findings: list
    highlighted_original: str
    highlighted_simplified: str
    summary: str
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "findings": [f.to_dict() for f in self.findings],
            "highlighted_original": self.highlighted_original,
            "highlighted_simplified": self.highlighted_simplified,
            "summary": self.summary,
            "disclaimer": self.disclaimer,
        }
