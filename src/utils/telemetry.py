"""
Telemetry utilities.

Provides Prometheus metrics and a FastAPI middleware to measure:
 - request count (method, path, status)
 - request latency histogram
 - request errors
 - model accuracy gauge (set externally on startup)
"""

from __future__ import annotations
import re
import time
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

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

_DIGIT_SEGMENT_RE = re.compile(r"^\d+$")
_UUID_SEGMENT_RE = re.compile(r"^[0-9a-fA-F-]{8,}$")


def _anonymize_path(path: str) -> str:
    """Fallback anonymisation when no route template is available."""
    if not path or path == "/":
        return "/"

    has_trailing_slash = path.endswith("/") and path != "/"
    segments = [seg for seg in path.strip("/").split("/") if seg]
    normalized_segments = []
    for segment in segments:
        if _DIGIT_SEGMENT_RE.match(segment) or _UUID_SEGMENT_RE.match(segment):
            normalized_segments.append("{id}")
        else:
            normalized_segments.append(segment)

    normalized_path = "/" + "/".join(normalized_segments)
    if has_trailing_slash and normalized_path != "/":
        normalized_path += "/"
    return normalized_path


def normalize_request_path(request: Request) -> str:
    """Return the route template if available, otherwise use anonymised path."""
    route = request.scope.get("route")
    if route is not None:
        for attr in ("path", "path_format"):
            template = getattr(route, attr, None)
            if template:
                return template
    return _anonymize_path(request.url.path)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """A Starlette middleware that collects Prometheus metrics for each request.

    This middleware tracks request counts, latency, and errors, and exposes
    them as Prometheus metrics.

    Args:
        app: The Starlette application instance.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        """Processes a request and collects metrics.

        This method times the request, captures its method, path, and status code,
        and increments the appropriate Prometheus counters and histograms.

        Args:
            request: The incoming Starlette request object.
            call_next: The next middleware or endpoint in the chain.

        Returns:
            The response from the next middleware or endpoint.
        """
        start = time.time()
        method = request.method
        normalized_path = None
        try:
            resp = await call_next(request)
            status_code = resp.status_code
        except Exception as exc:
            # Record errors as 500
            normalized_path = normalize_request_path(request)
            REQUEST_ERRORS.labels(method=method, path=normalized_path).inc()
            REQUEST_COUNT.labels(method=method, path=normalized_path, status="500").inc()
            raise
        finally:
            # record latency for all requests (success and exception)
            if normalized_path is None:
                normalized_path = normalize_request_path(request)
            latency = time.time() - start
            REQUEST_LATENCY.labels(method=method, path=normalized_path).observe(latency)

        # record remaining metrics (not in finally to avoid duplication if exception occurs)
        REQUEST_COUNT.labels(method=method, path=normalized_path, status=str(status_code)).inc()
        if 500 <= status_code < 600:
            REQUEST_ERRORS.labels(method=method, path=normalized_path).inc()
        return resp

def metrics_response() -> Response:
    """Generates a Prometheus metrics response.

    This function collects the latest metrics from the Prometheus registry
    and returns them in the Prometheus text format.

    Returns:
        A Starlette Response object containing the Prometheus metrics.
    """
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
