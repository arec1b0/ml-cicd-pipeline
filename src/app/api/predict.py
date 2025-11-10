"""
Prediction endpoints.
Defines a small, strict API surface suitable for CI and integration tests.

Comments and docstrings are written in English per repo standard.
"""

from __future__ import annotations
from typing import List
import logging
import os
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.app.config import MAX_BATCH_SIZE
from src.utils.drift import emit_prediction_log
from src.utils.tracing import get_tracer
from src.data.feature_statistics import get_feature_statistics, validate_feature_range

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])
limiter = Limiter(key_func=get_remote_address)

# Feature dimension limit - default to 4 for Iris dataset
# This can be overridden by environment variable or dynamically set from model metadata
EXPECTED_FEATURE_DIMENSION = int(os.getenv("EXPECTED_FEATURE_DIMENSION", "4"))

# Input schema: list of numeric feature vectors.
class PredictRequest(BaseModel):
    """Request model for the /predict endpoint.

    Attributes:
        features: A batch of feature vectors, where each feature vector is a list of floats.
                  Batch size must not exceed MAX_BATCH_SIZE, and each feature vector must
                  match the model's expected input dimension (default: 4 for Iris dataset).
    """
    features: List[List[float]] = Field(..., description="A batch of input features")

class PredictResponse(BaseModel):
    """Response model for the /predict endpoint.

    Attributes:
        predictions: A list of integer predictions.
    """
    predictions: List[int]

@router.post("/", response_model=PredictResponse)
@limiter.limit("100/minute")
async def predict(request: Request, payload: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """Runs model inference on a batch of feature vectors.

    This endpoint validates the input payload (batch size and feature dimensions),
    checks for model readiness, and uses the loaded model to generate predictions.
    It also logs prediction metadata and emits telemetry.

    Args:
        request: The incoming FastAPI request object.
        payload: The request body, containing the feature vectors for inference.
        background_tasks: FastAPI's background task runner.

    Raises:
        HTTPException: If the model is not loaded (503), the input is invalid (400),
                       or an internal error occurs during prediction (500).

    Returns:
        PredictResponse: An object containing the list of predictions.
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

    # Get the expected feature dimension from app state (set at startup from model)
    # Falls back to global EXPECTED_FEATURE_DIMENSION if not set
    expected_dim = getattr(app.state, "expected_feature_dimension", EXPECTED_FEATURE_DIMENSION)

    # Validate batch size
    batch_size = len(payload.features)
    if batch_size == 0:
        logger.warning(
            "Empty batch received",
            extra={"correlation_id": correlation_id}
        )
        raise HTTPException(status_code=400, detail="Input features cannot be empty")
    
    if batch_size > MAX_BATCH_SIZE:
        logger.warning(
            f"Batch size {batch_size} exceeds maximum of {MAX_BATCH_SIZE}",
            extra={"correlation_id": correlation_id, "batch_size": batch_size, "max_batch_size": MAX_BATCH_SIZE}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {batch_size} exceeds maximum of {MAX_BATCH_SIZE}"
        )

    # Validate feature dimensions: each item must be a non-empty list of floats with correct dimension
    for i, row in enumerate(payload.features):
        if not isinstance(row, list) or len(row) == 0:
            logger.warning(
                f"Invalid feature shape at index {i}",
                extra={"correlation_id": correlation_id, "feature_index": i}
            )
            raise HTTPException(status_code=400, detail=f"features[{i}] must be a non-empty list of numbers")

        if len(row) != expected_dim:
            logger.warning(
                f"Invalid feature dimension at index {i}",
                extra={
                    "correlation_id": correlation_id,
                    "feature_index": i,
                    "expected_dimension": expected_dim,
                    "actual_dimension": len(row)
                }
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature dimension at index {i}. Expected {expected_dim}, got {len(row)}"
            )
        
        # Validate feature values: check for NaN and infinity
        for j, value in enumerate(row):
            try:
                # Check for NaN and infinity
                if not np.isfinite(value):
                    logger.warning(
                        f"Invalid value in features[{i}][{j}]",
                        extra={
                            "correlation_id": correlation_id,
                            "feature_index": i,
                            "value_index": j,
                            "value": value
                        }
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value in features[{i}][{j}]: {value} (NaN or infinity not allowed)"
                    )
            except (TypeError, ValueError):
                # Handle cases where value cannot be converted to float
                logger.warning(
                    f"Non-numeric value in features[{i}][{j}]",
                    extra={
                        "correlation_id": correlation_id,
                        "feature_index": i,
                        "value_index": j,
                        "value": value,
                        "type": type(value).__name__
                    }
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value in features[{i}][{j}]: {value} (must be numeric)"
                )

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
