"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LANG = re.compile(r"^[a-z]{2,3}([-_][a-z0-9]{2,8})?$")


# ------------------------------------------------------------------ requests
class PairIn(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled field is an error, not silently ignored

    original_text: str
    simplified_text: str
    language: str = "en"
    context: str | None = None
    legal_domain: str | None = Field(default=None, max_length=100)

    @field_validator("original_text", "simplified_text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace-only")
        return v

    @field_validator("language")
    @classmethod
    def _language_code(cls, v: str) -> str:
        v = v.strip().lower()
        if not _LANG.match(v):
            raise ValueError("must be a language code such as 'en'")
        return re.split(r"[-_]", v)[0]  # 'en-IN' -> 'en'


class PredictRequest(PairIn):
    include_baselines: bool = False


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PairIn] = Field(min_length=1)
    include_baselines: bool = False


# ------------------------------------------------------------------ shared pieces
class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None
    supported_languages: list[str] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ComponentStatus(BaseModel):
    available: bool
    reason: str | None = None
    message: str | None = None


class ModelBrief(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    available: bool
    backbone: str
    version: str | None = None
    reason: str | None = None
    message: str | None = None


class SpanOut(BaseModel):
    text: str
    start: int
    end: int


class FindingOut(BaseModel):
    error_type: str
    change_kind: str
    original_span: SpanOut | None = None
    simplified_span: SpanOut | None = None
    explanation: str
    evidence_strength: str
    rule_id: str


class BaselineScore(BaseModel):
    status: Literal["available", "unavailable"]
    score: float | None = None
    raw_similarity: float | None = None
    reason: str | None = None
    message: str | None = None


class BaselineResults(BaseModel):
    requested: bool
    scores: dict[str, BaselineScore] = Field(default_factory=dict)


# ------------------------------------------------------------------ responses
class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    language: str
    lmp_status: Literal["available", "unavailable"]
    predicted_lmp_score: float | None = None
    preservation_category: str | None = None
    lmp_unavailable_reason: str | None = None
    legal_error_findings: list[FindingOut] = Field(default_factory=list)
    highlighted_original: str | None = None
    highlighted_simplified: str | None = None
    explanation_summary: str | None = None
    explanation_status: ComponentStatus
    baseline_scores: BaselineResults
    model: ModelBrief
    model_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class BatchItemResult(BaseModel):
    index: int
    status: Literal["ok", "error"]
    result: PredictResponse | None = None
    error: ErrorBody | None = None


class BatchResponse(BaseModel):
    count: int
    ok_count: int
    error_count: int
    lmp_status: Literal["available", "unavailable"]
    results: list[BatchItemResult]


class ExplanationHealth(BaseModel):
    available: bool
    supported_languages: list[str]
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ready_for_lmp_prediction: bool
    model: ModelBrief
    explanations: ExplanationHealth
    baselines: dict[str, ComponentStatus]


class BaselineInfo(BaseModel):
    available: bool
    reason: str | None = None
    message: str | None = None
    metadata: dict | None = None


class EvaluationInfo(BaseModel):
    status: Literal["saved_results", "no_saved_results", "unreadable"]
    source: str | None = None
    metrics: dict | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backbone: str
    model: ModelBrief
    supported_languages: list[str]
    planned_languages_not_supported: list[str]
    score_range: list[float]
    category_thresholds: dict[str, float]
    baselines: dict[str, BaselineInfo]
    evaluation: EvaluationInfo
    disclaimer: str
