"""Orchestration: one pipeline used by /predict and /batch-predict.

Reuses (does not reimplement) app.explanations.explain and app.scoring.categorize. An LMP score only ever comes
from a Predictor; it is never derived from rule-based findings or defaulted.
"""
from __future__ import annotations
import json
import logging

from .baseline_service import BaselineRegistry
from .config import Settings
from .errors import ApiError
from .explanations import DISCLAIMER, SUPPORTED_LANGUAGES, explain
from .inference import InferenceError, ModelStatus, Predictor, validate_scores
from .schemas import (BaselineInfo, BaselineResults, ComponentStatus, EvaluationInfo, ExplanationHealth, FindingOut,
                      HealthResponse, ModelBrief, ModelInfoResponse, PairIn, PredictResponse)
from .scoring import SCORE_MAX, SCORE_MIN, categorize

log = logging.getLogger("judgebert.api")
LIMITATIONS = (
    "Research prototype; not legal advice and not a substitute for a qualified legal professional.",
    DISCLAIMER,
    "Rule-based findings never influence the LMP score.",
    "Category thresholds are proposed defaults, not a legal standard.",
)
PLANNED_UNSUPPORTED = ["hi"]


class AnalysisService:
    def __init__(self, settings: Settings, predictor: Predictor | None, model_status: ModelStatus,
                 baselines: BaselineRegistry):
        self.settings, self.predictor, self.model_status, self.baselines = settings, predictor, model_status, baselines
        self.preserved_min, self.partial_min = settings.thresholds()
        self.supported_languages = list(SUPPORTED_LANGUAGES)
        self.explanations_ok, self.explanations_reason = True, None
        try:  # self-check once at startup
            explain("The tenant must pay.", "The tenant may pay.", "en")
        except Exception:
            log.exception("explanation module self-check failed")
            self.explanations_ok, self.explanations_reason = False, "self_check_failed"

    # ------------------------------------------------------------ status endpoints
    def model_brief(self) -> ModelBrief:
        s = self.model_status
        return ModelBrief(available=s.available, backbone=s.backbone, version=s.version,
                          reason=s.reason, message=s.message)

    @property
    def lmp_available(self) -> bool:
        return self.predictor is not None and self.model_status.available

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok", ready_for_lmp_prediction=self.lmp_available, model=self.model_brief(),
            explanations=ExplanationHealth(available=self.explanations_ok, supported_languages=self.supported_languages,
                                           reason=self.explanations_reason),
            baselines={k: ComponentStatus(available=v.available, reason=v.reason, message=v.message)
                       for k, v in self.baselines.infos.items()})

    def _evaluation(self) -> EvaluationInfo:
        path = self.settings.results_dir / "metrics.json"
        if not path.is_file():
            return EvaluationInfo(status="no_saved_results")
        try:
            if path.stat().st_size > 1_000_000:
                raise ValueError
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            return EvaluationInfo(status="unreadable", source=path.name)
        return EvaluationInfo(status="saved_results", source=path.name, metrics=data)

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            backbone=self.settings.backbone, model=self.model_brief(), supported_languages=self.supported_languages,
            planned_languages_not_supported=PLANNED_UNSUPPORTED, score_range=[SCORE_MIN, SCORE_MAX],
            category_thresholds={"preserved_min": self.preserved_min, "partial_min": self.partial_min},
            baselines=dict(self.baselines.infos), evaluation=self._evaluation(),
            disclaimer=LIMITATIONS[0])

    # ------------------------------------------------------------ analysis
    def _check_item(self, it: PairIn) -> ApiError | None:
        m = self.settings.max_chars
        for name in ("original_text", "simplified_text", "context"):
            v = getattr(it, name)
            if v is not None and len(v) > m:
                return ApiError(422, "input_too_long", f"'{name}' is longer than {m} characters.")
        if it.language not in self.supported_languages:
            return ApiError(422, "unsupported_language",
                            f"Language '{it.language}' is not supported. Supported: {', '.join(self.supported_languages)}.",
                            supported_languages=self.supported_languages)
        return None

    def _scores(self, items: list[PairIn]) -> list[float] | None:
        if not self.lmp_available or not items:
            return None
        try:
            raw = self.predictor.predict([i.original_text for i in items], [i.simplified_text for i in items],
                                         [i.language for i in items])
            return validate_scores(raw, len(items))
        except Exception:  # InferenceError or any model failure; details go to the log, not the client
            log.exception("LMP inference failed")
            raise ApiError(500, "inference_failed", "Model inference failed.") from None

    def analyze(self, items: list[PairIn], include_baselines: bool = False) -> list:
        """Return, per item, a PredictResponse or an ApiError (not raised) for per-item problems."""
        results: list = [None] * len(items)
        valid: list[int] = []
        for i, it in enumerate(items):
            err = self._check_item(it)
            if err is not None:
                results[i] = err
            else:
                valid.append(i)
        scores = self._scores([items[i] for i in valid])  # one predictor call for the whole request
        for pos, i in enumerate(valid):
            results[i] = self._build(items[i], None if scores is None else scores[pos], include_baselines)
        return results

    def _build(self, it: PairIn, score: float | None, include_baselines: bool) -> PredictResponse:
        warnings: list[str] = []
        category = categorize(score, self.preserved_min, self.partial_min) if score is not None else None
        if score is None:
            warnings.append(f"No LMP score was produced: {self.model_status.message}")
        if it.context:
            warnings.append("'context' was accepted but is not used by the current pipeline.")
        if it.legal_domain:
            warnings.append("'legal_domain' was accepted but is not used by the current pipeline.")

        findings, ho, hs, summary = [], None, None, None
        try:
            r = explain(it.original_text, it.simplified_text, it.language)
            findings = [FindingOut.model_validate(f.to_dict()) for f in r.findings]
            ho, hs, summary = r.highlighted_original, r.highlighted_simplified, r.summary
            expl = ComponentStatus(available=True)
        except Exception:
            log.exception("explanation module failed")
            expl = ComponentStatus(available=False, reason="explanation_error", message="Explanations failed for this pair.")
            warnings.append("Explanations could not be produced for this pair.")

        base = BaselineResults(requested=include_baselines)
        if include_baselines:
            base.scores = {n: self.baselines.score_pair(n, it.original_text, it.simplified_text, it.language)
                           for n in self.baselines.infos}
            if not any(b.status == "available" for b in base.scores.values()):
                warnings.append("Baseline scores were requested but no fitted baseline is available.")

        return PredictResponse(
            language=it.language, lmp_status="available" if score is not None else "unavailable",
            predicted_lmp_score=score, preservation_category=category,
            lmp_unavailable_reason=None if score is not None else self.model_status.reason,
            legal_error_findings=findings, highlighted_original=ho, highlighted_simplified=hs,
            explanation_summary=summary, explanation_status=expl, baseline_scores=base, model=self.model_brief(),
            model_version=self.model_status.version if score is not None else None,
            warnings=warnings, limitations=list(LIMITATIONS))
