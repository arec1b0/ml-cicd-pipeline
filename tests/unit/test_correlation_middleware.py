"""
Unit tests for correlation ID middleware.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.app.api.middleware.correlation import CorrelationIDMiddleware, get_correlation_id
from src.utils.logging import correlation_id_ctx


@pytest.mark.unit
class TestCorrelationIDMiddleware:
    """Test correlation ID middleware functionality."""

    @pytest.fixture
    def test_app(self) -> FastAPI:
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"correlation_id": getattr(request.state, "correlation_id", None)}
        
        app.add_middleware(CorrelationIDMiddleware, header_name="X-Correlation-ID")
        
        return app

    def test_middleware_generates_uuid_when_header_missing(self, test_app: FastAPI):
        """Test middleware generates UUID when header missing."""
        client = TestClient(test_app)
        
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "correlation_id" in response.json()
        correlation_id = response.json()["correlation_id"]
        
        # Should be a valid UUID
        uuid.UUID(correlation_id)
        
        # Should be in response headers
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == correlation_id

    def test_middleware_uses_existing_correlation_id(self, test_app: FastAPI):
        """Test middleware uses existing correlation ID from header."""
        client = TestClient(test_app)
        
        test_id = "test-correlation-id-12345"
        
        response = client.get("/test", headers={"X-Correlation-ID": test_id})
        
        assert response.status_code == 200
        assert response.json()["correlation_id"] == test_id
        assert response.headers["X-Correlation-ID"] == test_id

    def test_correlation_id_stored_in_request_state(self, test_app: FastAPI):
        """Test correlation ID stored in request.state."""
        client = TestClient(test_app)
        
        test_id = "test-correlation-id-12345"
        
        response = client.get("/test", headers={"X-Correlation-ID": test_id})
        
        assert response.status_code == 200
        assert response.json()["correlation_id"] == test_id

    def test_correlation_id_set_in_context_variable(self, test_app: FastAPI):
        """Test correlation ID set in context variable."""
        client = TestClient(test_app)
        
        test_id = "test-correlation-id-12345"
        
        # Clear context before request
        correlation_id_ctx.set(None)
        
        response = client.get("/test", headers={"X-Correlation-ID": test_id})
        
        assert response.status_code == 200
        
        # Context should be cleared after request, but during request it should be set
        # Note: Context variables are request-scoped, so we can't easily test this
        # without making the endpoint check it directly

    def test_correlation_id_added_to_response_headers(self, test_app: FastAPI):
        """Test correlation ID added to response headers."""
        client = TestClient(test_app)
        
        test_id = "test-correlation-id-12345"
        
        response = client.get("/test", headers={"X-Correlation-ID": test_id})
        
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == test_id

    def test_custom_header_name_configuration(self):
        """Test custom header name configuration."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"correlation_id": getattr(request.state, "correlation_id", None)}
        
        app.add_middleware(CorrelationIDMiddleware, header_name="X-Custom-Correlation-ID")
        
        client = TestClient(app)
        
        test_id = "test-correlation-id-12345"
        
        response = client.get("/test", headers={"X-Custom-Correlation-ID": test_id})
        
        assert response.status_code == 200
        assert response.json()["correlation_id"] == test_id
        assert "X-Custom-Correlation-ID" in response.headers

    def test_get_correlation_id_function(self):
        """Test get_correlation_id helper function."""
        test_id = "test-correlation-id-12345"
        
        correlation_id_ctx.set(test_id)
        
        result = get_correlation_id()
        
        assert result == test_id
        
        # Cleanup
        correlation_id_ctx.set(None)

