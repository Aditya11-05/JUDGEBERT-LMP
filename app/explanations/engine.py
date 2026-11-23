"""Public entry point: explain(original, simplified, language) -> ExplanationResult."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from . import lexicon_en
from .alignment import mark, to_html
from .rules import ALL_RULES
from .types import STRENGTHS, ExplanationResult, Features, Finding

MAX_CHARS = 20_000


class UnsupportedLanguageError(ValueError):
    """Raised when no rule set has been implemented AND tested for the language."""


@dataclass(frozen=True)
class RuleSet:
    language: str
    extract: Callable[[str], Features]
    entity_eq: Callable


# Only languages with implemented extractors AND unit tests belong here. Hindi is NOT registered yet.
RULESETS = {"en": RuleSet("en", lexicon_en.extract, lexicon_en.entity_eq)}
SUPPORTED_LANGUAGES = tuple(RULESETS)


def _check(name: str, text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(text) > MAX_CHARS:
        raise ValueError(f"{name} is longer than {MAX_CHARS} characters")
    return text


def find_changes(original: str, simplified: str, language: str = "en", min_strength: str = "weak") -> list[Finding]:
    _check("original", original)
    _check("simplified", simplified)
    if min_strength not in STRENGTHS:
        raise ValueError(f"min_strength must be one of {STRENGTHS}")
    rs = RULESETS.get(language)
    if rs is None:
        raise UnsupportedLanguageError(
            f"No tested rule set for language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}.")
    fo, fs = rs.extract(original), rs.extract(simplified)
    findings = [f for rule in ALL_RULES for f in rule(fo, fs, rs)]
    for f in findings:
        f.language = language
    limit = STRENGTHS.index(min_strength)
    findings = [f for f in findings if STRENGTHS.index(f.evidence_strength) <= limit]
    findings.sort(key=lambda f: (STRENGTHS.index(f.evidence_strength),
                                 f.original_span.start if f.original_span else 10**9,
                                 f.simplified_span.start if f.simplified_span else 10**9, f.rule_id))
    return findings


def explain(original: str, simplified: str, language: str = "en", min_strength: str = "weak") -> ExplanationResult:
    findings = find_changes(original, simplified, language, min_strength)
    ho = mark(original, [f.original_span for f in findings])
    hs = mark(simplified, [f.simplified_span for f in findings])
    if findings:
        types = sorted({f.error_type for f in findings})
        summary = (f"{len(findings)} potential change(s) flagged ({', '.join(types)}). These are heuristic hints; "
                   "review the highlighted phrases manually.")
    else:
        summary = ("No potential legal-meaning change was detected by the rule set. This does not show that the meaning is "
                   "preserved: the rules only cover the listed categories.")
    return ExplanationResult(language, findings, ho, hs, summary)


def highlight_html(original: str, simplified: str, findings: list[Finding]) -> tuple[str, str]:
    return (to_html(original, [f.original_span for f in findings]),
            to_html(simplified, [f.simplified_span for f in findings]))
