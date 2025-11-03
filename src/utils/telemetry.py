"""
Telemetry utilities.

Provides Prometheus metrics and a FastAPI middleware to measure:
 - request count (method, path, status)
 - request latency histogram
 - request errors
 - model accuracy gauge (set externally on startup)
"""

from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
from typing import Callable

REQUEST_COUNT = Counter(
    "ml_request_count",
    "Total HTTP requests processed",
    ["method", "path", "status"]
)

REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Request latency in seconds",
    ["method", "path"]
)

REQUEST_ERRORS = Counter(
    "ml_request_errors_total",
    "Total HTTP 5xx responses",
    ["method", "path"]
)

MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "Current model accuracy as reported by trainer (0.0-1.0)"
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that collects Prometheus metrics per request.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.time()
        method = request.method
        path = request.url.path
        try:
            resp = await call_next(request)
            status_code = resp.status_code
        except Exception as exc:
            # Record errors as 500
            REQUEST_ERRORS.labels(method=method, path=path).inc()
            REQUEST_COUNT.labels(method=method, path=path, status="500").inc()
            raise
        finally:
            # record latency for all requests (success and exception)
            latency = time.time() - start
            REQUEST_LATENCY.labels(method=method, path=path).observe(latency)

        # record remaining metrics (not in finally to avoid duplication if exception occurs)
        REQUEST_COUNT.labels(method=method, path=path, status=str(status_code)).inc()
        if 500 <= status_code < 600:
            REQUEST_ERRORS.labels(method=method, path=path).inc()
        return resp

def metrics_response() -> Response:
    """
    Return the current metrics in Prometheus text format.
    """
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
