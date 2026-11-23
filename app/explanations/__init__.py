"""Rule-based legal-error explanations (English implemented; Hindi planned, NOT yet supported)."""
from .engine import (SUPPORTED_LANGUAGES, UnsupportedLanguageError, explain, find_changes,  # noqa: F401
                     highlight_html)
from .types import (DISCLAIMER, ERROR_TYPES, HUMAN_ONLY_TYPES, STRENGTHS, ExplanationResult,  # noqa: F401
                    Finding, Span)
