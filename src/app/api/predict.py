"""
Prediction endpoints.
Defines a small, strict API surface suitable for CI and integration tests.

Comments and docstrings are written in English per repo standard.
"""

from __future__ import annotations
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from src.utils.drift import emit_prediction_log

router = APIRouter(prefix="/predict", tags=["predict"])

# Input schema: list of numeric feature vectors.
class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]

@router.post("/", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """
    Predict endpoint expects JSON body: {"features": [[...], [...]]}.
    The model instance is attached to app.state.ml_wrapper during startup.
    This handler validates input shape and returns integer-like predictions
    as ints for compact JSON.
    """
    app = request.app

    # Check readiness flag and wrapper placed by startup hook
    model_wrapper = getattr(app.state, "ml_wrapper", None)
    is_ready = bool(getattr(app.state, "is_ready", False))
    if not is_ready or model_wrapper is None:
        # Return clear error for orchestrator / CI to act on
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate shape: each item must be a non-empty list of floats
    for i, row in enumerate(payload.features):
        if not isinstance(row, list) or len(row) == 0:
            raise HTTPException(status_code=400, detail=f"features[{i}] must be a non-empty list of numbers")

    try:
        preds = model_wrapper.predict(payload.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Normalize integer-like floats to ints
    normalized: list = []
    for p in preds:
        if isinstance(p, float) and p.is_integer():
            normalized.append(int(p))
        else:
            normalized.append(p)

    metadata = {
        "path": str(request.url.path),
        "client_host": getattr(request.client, "host", None),
        "model": getattr(app.state, "model_metadata", None),
        "headers": {
            "user_agent": request.headers.get("user-agent"),
            "x_request_id": request.headers.get("x-request-id"),
        },
    }
    background_tasks.add_task(
        emit_prediction_log,
        features=payload.features,
        predictions=normalized,
        metadata=metadata,
    )
    return PredictResponse(predictions=normalized)
