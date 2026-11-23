"""FastAPI application factory. Run: uvicorn app.main:app"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .baseline_service import BaselineRegistry
from .config import Settings
from .errors import ApiError
from .inference import ModelStatus, Predictor, load_predictor
from .routes import router
from .service import AnalysisService

log = logging.getLogger("judgebert.api")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None,
               baselines: BaselineRegistry | None = None) -> FastAPI:
    """`predictor` / `baselines` may be injected (tests, or a trained model later); otherwise they are loaded from disk."""
    cfg = settings or Settings.from_env()
    cfg.thresholds()  # fail fast on invalid configuration

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            pred, status = predictor, ModelStatus(True, None, "Predictor supplied at startup.", predictor.version,
                                                  predictor.backbone)
        else:
            try:
                pred, status = load_predictor(cfg)  # once, at startup
            except Exception:
                log.exception("model loading failed")
                pred, status = None, ModelStatus(False, "load_error", "The model failed to load.", None, cfg.backbone)
        app.state.service = AnalysisService(cfg, pred, status, baselines or BaselineRegistry.load(cfg))
        yield
        app.state.service = None

    app = FastAPI(title="JUDGEBERT-LMP API", version="0.1.0", lifespan=lifespan,
                  description="Research prototype for legal meaning preservation. Not legal advice.")

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > cfg.max_body_bytes:
            return JSONResponse(ApiError(413, "payload_too_large", "Request body is too large.").body(), status_code=413)
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse(e.body(), status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in err["loc"] if p != "body") or "body",
                    "message": str(err["msg"])} for err in e.errors()]
        body = ApiError(422, "validation_error", "The request is invalid.", details=details).body()
        return JSONResponse(body, status_code=422)  # input values are deliberately not echoed back

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
        return JSONResponse(ApiError(e.status_code, code, "Request could not be served.").body(), status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")  # stack trace stays in the server log
        return JSONResponse(ApiError(500, "internal_error", "Internal server error.").body(), status_code=500)

    app.include_router(router)
    return app


app = create_app()
