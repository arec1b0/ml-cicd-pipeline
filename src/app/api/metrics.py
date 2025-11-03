"""
/metrics endpoint - exposes Prometheus metrics collected by telemetry middleware.
"""

from __future__ import annotations
from fastapi import APIRouter
from src.utils.telemetry import metrics_response

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/", include_in_schema=False)
async def metrics():
    """Exposes Prometheus metrics.

    This endpoint is scraped by Prometheus to collect application and model
    performance metrics.

    Returns:
        A Starlette Response object containing the Prometheus metrics.
    """
    return metrics_response()
