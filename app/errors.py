"""API error type. Bodies are always {"error": {...}} and never contain stack traces or filesystem paths."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: list | None = None,
                 supported_languages: list[str] | None = None):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.details, self.supported_languages = details, supported_languages

    def body(self) -> dict:
        err: dict = {"code": self.code, "message": self.message}
        if self.details:
            err["details"] = self.details
        if self.supported_languages is not None:
            err["supported_languages"] = self.supported_languages
        return {"error": err}
