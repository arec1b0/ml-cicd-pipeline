"""
Prediction endpoints.
Defines a small, strict API surface suitable for CI and integration tests.

Comments and docstrings are written in English per repo standard.
"""

from __future__ import annotations
from typing import List
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from src.utils.drift import emit_prediction_log
from src.utils.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

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
    correlation_id = getattr(request.state, "correlation_id", None)

    # Check readiness flag and wrapper placed by startup hook
    model_wrapper = getattr(app.state, "ml_wrapper", None)
    is_ready = bool(getattr(app.state, "is_ready", False))
    if not is_ready or model_wrapper is None:
        # Return clear error for orchestrator / CI to act on
        logger.error("Model not loaded", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate shape: each item must be a non-empty list of floats
    for i, row in enumerate(payload.features):
        if not isinstance(row, list) or len(row) == 0:
            logger.warning(
                f"Invalid feature shape at index {i}",
                extra={"correlation_id": correlation_id, "feature_index": i}
            )
            raise HTTPException(status_code=400, detail=f"features[{i}] must be a non-empty list of numbers")

    logger.info(
        "Processing prediction request",
        extra={
            "correlation_id": correlation_id,
            "feature_count": len(payload.features),
            "feature_dim": len(payload.features[0]) if payload.features else 0
        }
    )

    # Get model metadata for span attributes
    model_metadata = getattr(app.state, "model_metadata", None)
    model_path = model_metadata.get("model_path") if model_metadata else None

    try:
        # Create custom span for model inference
        with tracer.start_as_current_span("model_inference") as span:
            # Add span attributes
            span.set_attribute("ml.model.path", model_path or "unknown")
            span.set_attribute("ml.input.feature_count", len(payload.features))
            span.set_attribute("ml.input.feature_dim", len(payload.features[0]) if payload.features else 0)
            if correlation_id:
                span.set_attribute("correlation.id", correlation_id)
            
            # Execute model prediction within span
            preds = model_wrapper.predict(payload.features)
            
            # Add output attributes
            span.set_attribute("ml.output.prediction_count", len(preds))
            
            logger.info(
                "Prediction completed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "prediction_count": len(preds)
                }
            )
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"correlation_id": correlation_id, "error": str(exc)}
        )
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
