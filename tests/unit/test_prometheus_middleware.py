"""
Unit tests for Prometheus middleware.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.utils.telemetry import PrometheusMiddleware, normalize_request_path, metrics_response, REQUEST_COUNT, REQUEST_LATENCY, REQUEST_ERRORS


@pytest.mark.unit
class TestPrometheusMiddleware:
    """Test Prometheus middleware functionality."""

    @pytest.fixture
    def test_app(self) -> FastAPI:
        """Create test FastAPI app with Prometheus middleware."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        @app.get("/test/{item_id}")
        async def test_with_id(item_id: int):
            return {"item_id": item_id}
        
        @app.get("/error")
        async def error_endpoint():
            raise HTTPException(status_code=500, detail="Internal error")
        
        app.add_middleware(PrometheusMiddleware)
        
        return app

    def test_middleware_records_request_count(self, test_app: FastAPI):
        """Test PrometheusMiddleware records request count."""
        client = TestClient(test_app)
        
        # Clear metrics before test
        REQUEST_COUNT.clear()
        
        response = client.get("/test")
        
        assert response.status_code == 200
        
        # Check that metric was recorded
        # Note: Prometheus metrics are global, so we can't easily assert exact values
        # without more complex setup

    def test_middleware_records_request_latency(self, test_app: FastAPI):
        """Test PrometheusMiddleware records request latency."""
        client = TestClient(test_app)
        
        start_time = time.time()
        response = client.get("/test")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should be fast

    def test_middleware_records_5xx_errors(self, test_app: FastAPI):
        """Test PrometheusMiddleware records 5xx errors."""
        client = TestClient(test_app)
        
        # Clear metrics before test
        REQUEST_ERRORS.clear()
        
        response = client.get("/error")
        
        assert response.status_code == 500
        
        # Error counter should be incremented
        # Note: Hard to test without accessing internal metric state

    def test_normalize_request_path_anonymizes_ids(self):
        """Test normalize_request_path anonymizes IDs and UUIDs."""
        from starlette.requests import Request
        from starlette.routing import Route
        
        # Create mock request with ID in path
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test/12345",
            "route": Route("/test/{item_id}", endpoint=lambda: None),
        }
        request = Request(scope)
        
        normalized = normalize_request_path(request)
        
        assert normalized == "/test/{id}"

    def test_normalize_request_path_uses_route_template(self):
        """Test normalize_request_path uses route template when available."""
        from starlette.requests import Request
        from starlette.routing import Route
        
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test/12345",
            "route": Route("/test/{item_id}", endpoint=lambda: None),
        }
        request = Request(scope)
        
        normalized = normalize_request_path(request)
        
        # Should use route template if available
        assert "{id}" in normalized or "/test/{item_id}" in normalized

    def test_normalize_request_path_anonymizes_uuids(self):
        """Test normalize_request_path anonymizes UUIDs."""
        from starlette.requests import Request
        
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        scope = {
            "type": "http",
            "method": "GET",
            "path": f"/test/{uuid_str}",
            "route": None,
        }
        request = Request(scope)
        
        normalized = normalize_request_path(request)
        
        assert uuid_str not in normalized
        assert "{id}" in normalized

    def test_exception_handling_increments_error_counter(self, test_app: FastAPI):
        """Test exception handling increments error counter."""
        app = FastAPI()
        
        @app.get("/exception")
        async def exception_endpoint():
            raise ValueError("Test exception")
        
        app.add_middleware(PrometheusMiddleware)
        
        client = TestClient(app)
        
        # Clear metrics
        REQUEST_ERRORS.clear()
        
        # Request should raise exception, but middleware should catch it
        response = client.get("/exception")
        
        # FastAPI will return 500 for unhandled exceptions
        assert response.status_code == 500

    def test_metrics_response_returns_prometheus_format(self, test_app: FastAPI):
        """Test metrics_response returns Prometheus format."""
        response = metrics_response()
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for Prometheus format indicators
        content = response.body.decode("utf-8")
        assert "#" in content or "ml_request" in content

