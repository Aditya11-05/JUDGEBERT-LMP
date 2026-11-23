"""HTTP client for the JUDGEBERT-LMP FastAPI backend. Real HTTP only; no backend internals are imported."""
from __future__ import annotations
import os

import httpx

from .schema_checks import InvalidResponse, check_batch, check_health, check_model_info, check_predict

ENV_URL = "JUDGEBERT_API_URL"
DEFAULT_API_URL = "http://localhost:8000"
MAX_BATCH_ITEMS = 100  # the backend's default limit (JUDGEBERT_MAX_BATCH)
DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=15.0, pool=3.0)


def api_url_from_env() -> str:
    return (os.environ.get(ENV_URL, "").strip() or DEFAULT_API_URL).rstrip("/")


class ApiClientError(Exception):
    """kind: 'offline' | 'timeout' | 'http' | 'invalid_response'. Messages are safe to show to users."""

    def __init__(self, kind: str, message: str, status: int | None = None, code: str | None = None,
                 details: list | None = None, supported_languages: list | None = None):
        super().__init__(message)
        self.kind, self.message, self.status, self.code = kind, message, status, code
        self.details, self.supported_languages = details or [], supported_languages


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or api_url_from_env()).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"{ENV_URL} must start with http:// or https:// (got {self.base_url!r})")
        self.timeout, self._transport = timeout, transport

    # ------------------------------------------------------------ transport
    def _request(self, method: str, path: str, json=None):
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport) as c:
                r = c.request(method, path, json=json)
        except httpx.TimeoutException as e:
            if isinstance(e, httpx.ConnectTimeout):
                raise ApiClientError("offline", f"Could not connect to the backend at {self.base_url} (connection timed out).") from None
            raise ApiClientError("timeout", "The backend did not answer in time. Try again or use shorter input.") from None
        except httpx.HTTPError:
            raise ApiClientError("offline", f"Could not reach the backend at {self.base_url}. Is it running?") from None
        if r.status_code >= 400:
            raise self._http_error(r)
        try:
            return r.json()
        except ValueError:
            raise ApiClientError("invalid_response", "The backend returned a response that is not valid JSON.") from None

    @staticmethod
    def _http_error(r: httpx.Response) -> ApiClientError:
        err = None
        try:
            body = r.json()
            err = body.get("error") if isinstance(body, dict) else None
        except ValueError:
            pass
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            details = err.get("details") if isinstance(err.get("details"), list) else []
            langs = err.get("supported_languages") if isinstance(err.get("supported_languages"), list) else None
            return ApiClientError("http", err["message"], r.status_code, str(err.get("code", "")), details, langs)
        return ApiClientError("http", f"The backend returned HTTP {r.status_code}.", r.status_code)

    def _checked(self, fn, *args):
        try:
            return fn(*args)
        except InvalidResponse as e:
            raise ApiClientError("invalid_response", f"The backend response was not in the expected format ({e}).") from None

    # ------------------------------------------------------------ endpoints
    def health(self) -> dict:
        return self._checked(check_health, self._request("GET", "/health"))

    def model_info(self) -> dict:
        return self._checked(check_model_info, self._request("GET", "/model-info"))

    def predict(self, original_text: str, simplified_text: str, language: str = "en", context: str | None = None,
                legal_domain: str | None = None, include_baselines: bool = False) -> dict:
        body = {"original_text": original_text, "simplified_text": simplified_text, "language": language,
                "include_baselines": bool(include_baselines)}
        if context and context.strip():
            body["context"] = context
        if legal_domain and legal_domain.strip():
            body["legal_domain"] = legal_domain
        return self._checked(check_predict, self._request("POST", "/predict", body))

    def batch_predict(self, items: list[dict], include_baselines: bool = False) -> dict:
        if not 1 <= len(items) <= MAX_BATCH_ITEMS:
            raise ValueError(f"a batch request must contain 1-{MAX_BATCH_ITEMS} items")
        data = self._request("POST", "/batch-predict", {"items": items, "include_baselines": bool(include_baselines)})
        return self._checked(check_batch, data, len(items))
