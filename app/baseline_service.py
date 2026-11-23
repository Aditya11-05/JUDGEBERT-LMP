"""Baseline registry: loads fitted baseline artifacts once and scores pairs."""
from __future__ import annotations

import logging

from .config import Settings
from .models.baselines import (
    TfidfBaseline,
    MultilingualEmbeddingBaseline,
    load_tfidf,
    load_multilingual,
)
from .schemas import BaselineInfo, BaselineScore

log = logging.getLogger("judgebert.api")

NAMES = ("tfidf", "multilingual_embedding")

_META_PUBLIC = (
    "n_train",
    "languages",
    "mapping",
    "fit_scope",
    "description",
    "model_name",
)


class BaselineRegistry:
    def __init__(
        self,
        infos: dict[str, BaselineInfo],
        tfidf: TfidfBaseline | None = None,
        multilingual: MultilingualEmbeddingBaseline | None = None,
        metadata: dict | None = None,
    ):
        self.infos = infos
        self._tfidf = tfidf
        self._multilingual = multilingual
        self._metadata = metadata or {}

    @classmethod
    def load(cls, settings: Settings) -> "BaselineRegistry":
        infos = {}
        models = {"tfidf": None, "multilingual_embedding": None}
        metadata = {}

        loaders = {
            "tfidf": load_tfidf,
            "multilingual_embedding": load_multilingual,
        }

        for name, loader in loaders.items():
            try:
                model, meta = loader(settings.baseline_dir)
                models[name] = model
                metadata[name] = meta
                infos[name] = BaselineInfo(
                    available=True,
                    metadata={
                        k: meta[k]
                        for k in _META_PUBLIC
                        if k in meta
                    },
                )
            except FileNotFoundError:
                infos[name] = BaselineInfo(
                    available=False,
                    reason="artifact_missing",
                    message=f"No fitted {name} baseline artifact is installed.",
                )
            except Exception:
                log.exception("%s baseline artifact rejected", name)
                infos[name] = BaselineInfo(
                    available=False,
                    reason="artifact_invalid",
                    message=f"The {name} baseline artifact is invalid.",
                )

        return cls(
            infos=infos,
            tfidf=models["tfidf"],
            multilingual=models["multilingual_embedding"],
            metadata=metadata,
        )

    def score_pair(
        self,
        name: str,
        original: str,
        simplified: str,
        language: str,
    ) -> BaselineScore:
        info = self.infos.get(name)

        if info is None or not info.available:
            return BaselineScore(
                status="unavailable",
                reason=getattr(info, "reason", "unknown_baseline"),
                message=getattr(info, "message", None),
            )

        meta = self._metadata.get(name, {})

        if language not in meta.get("languages", []):
            return BaselineScore(
                status="unavailable",
                reason="language_not_covered",
                message=f"The baseline was not fitted on '{language}' data.",
            )

        model = (
            self._tfidf
            if name == "tfidf"
            else self._multilingual
            if name == "multilingual_embedding"
            else None
        )

        if model is None:
            return BaselineScore(
                status="unavailable",
                reason="model_not_loaded",
                message="The baseline model is not loaded.",
            )

        try:
            raw = float(
                model.raw_similarity([original], [simplified])[0]
            )
            score = float(
                model.predict([original], [simplified])[0]
            )
        except Exception:
            log.exception("%s baseline scoring failed", name)
            return BaselineScore(
                status="unavailable",
                reason="baseline_error",
                message="Baseline scoring failed.",
            )

        return BaselineScore(
            status="available",
            score=score,
            raw_similarity=raw,
        )
