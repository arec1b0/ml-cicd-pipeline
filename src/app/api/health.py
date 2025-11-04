"""
Health endpoints.
Provides liveness/readiness and optional model metrics.
"""

from __future__ import annotations
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    """Response model for the /health endpoint.

    Attributes:
        status: The overall status of the service (e.g., "ok").
        ready: A boolean indicating if the model is loaded and ready for inference.
        details: Optional dictionary containing additional details, such as model metrics.
    """
    status: str
    ready: bool
    details: dict | None = None
    mlflow: dict | None = None

@router.get("/", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Performs a health and readiness check.

    This endpoint is used to verify the service's liveness and readiness.
    It checks the `is_ready` flag in the application state and includes
    model metrics in the response if they are available.

    Args:
        request: The incoming FastAPI request object.

    Returns:
        HealthResponse: An object indicating the service's status and readiness.
    """
    app = request.app
    ready = bool(getattr(app.state, "is_ready", False))
    details = None
    metrics = getattr(app.state, "ml_metrics", None)
    if metrics is not None:
        details = {"metrics": metrics}
    connectivity = getattr(app.state, "mlflow_connectivity", None)
    return HealthResponse(status="ok", ready=ready, details=details, mlflow=connectivity)
