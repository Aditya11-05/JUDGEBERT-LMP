"""HTTP routes (thin: validation and serialisation only; logic lives in service.py)."""
from __future__ import annotations
from fastapi import APIRouter, Request

from .errors import ApiError
from .schemas import (BatchItemResult, BatchRequest, BatchResponse, ErrorBody, ErrorResponse, HealthResponse,
                      ModelInfoResponse, PredictRequest, PredictResponse)
from .service import AnalysisService

router = APIRouter()
_ERR = {422: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}}


def get_service(request: Request) -> AnalysisService:
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise ApiError(503, "service_unavailable", "The service is not ready.")
    return svc


@router.get("/health", response_model=HealthResponse, responses=_ERR)
def health(request: Request):
    return get_service(request).health()


@router.get("/model-info", response_model=ModelInfoResponse, responses=_ERR)
def model_info(request: Request):
    return get_service(request).model_info()


@router.post("/predict", response_model=PredictResponse, responses=_ERR)
def predict(body: PredictRequest, request: Request):
    svc = get_service(request)
    (res,) = svc.analyze([body], body.include_baselines)
    if isinstance(res, ApiError):
        raise res
    return res


@router.post("/batch-predict", response_model=BatchResponse, responses=_ERR)
def batch_predict(body: BatchRequest, request: Request):
    svc = get_service(request)
    if len(body.items) > svc.settings.max_batch:
        raise ApiError(422, "batch_too_large", f"A batch may contain at most {svc.settings.max_batch} items.")
    out = svc.analyze(body.items, body.include_baselines)
    rows = []
    for i, r in enumerate(out):
        if isinstance(r, ApiError):
            rows.append(BatchItemResult(index=i, status="error", error=ErrorBody(**r.body()["error"])))
        else:
            rows.append(BatchItemResult(index=i, status="ok", result=r))
    n_ok = sum(r.status == "ok" for r in rows)
    return BatchResponse(count=len(rows), ok_count=n_ok, error_count=len(rows) - n_ok,
                         lmp_status="available" if svc.lmp_available else "unavailable", results=rows)
