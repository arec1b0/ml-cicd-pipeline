"""
Health endpoints.
Provides liveness/readiness and optional model metrics.
"""

from __future__ import annotations
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])

class HealthResponse(BaseModel):
    status: str
    ready: bool
    details: dict | None = None
    mlflow: dict | None = None

@router.get("/", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Liveness and readiness combined endpoint.
    Reads readiness flag and optional metrics from app.state.
    """
    app = request.app
    ready = bool(getattr(app.state, "is_ready", False))
    details = None
    metrics = getattr(app.state, "ml_metrics", None)
    if metrics is not None:
        details = {"metrics": metrics}
    connectivity = getattr(app.state, "mlflow_connectivity", None)
    return HealthResponse(status="ok", ready=ready, details=details, mlflow=connectivity)
