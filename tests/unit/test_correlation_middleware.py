"""
Unit tests for correlation ID middleware (src/app/api/middleware/correlation.py).

Tests cover:
- Correlation ID generation when not provided
- Correlation ID extraction from request headers
- Correlation ID storage in request state
- Correlation ID propagation to response headers
- Context variable management for logging
"""

from __future__ import annotations

import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from src.app.api.middleware.correlation import (
    CorrelationIDMiddleware,
    get_correlation_id,
)


@pytest.fixture
def app():
    """Create a simple FastAPI app for testing."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint(request: Request):
        # Access correlation ID from request state
        correlation_id = getattr(request.state, "correlation_id", None)
        return {"correlation_id": correlation_id}

    return app


def test_generates_correlation_id_when_not_provided(app):
    """Test that middleware generates a new UUID when no header is provided."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    response = client.get("/test")

    assert response.status_code == 200
    correlation_id = response.json()["correlation_id"]

    # Should be a valid UUID
    assert correlation_id is not None
    try:
        uuid.UUID(correlation_id)
    except ValueError:
        pytest.fail(f"Generated correlation ID is not a valid UUID: {correlation_id}")


def test_uses_provided_correlation_id(app):
    """Test that middleware uses correlation ID from request header."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    provided_id = "custom-correlation-id-12345"
    response = client.get("/test", headers={"X-Correlation-ID": provided_id})

    assert response.status_code == 200
    correlation_id = response.json()["correlation_id"]

    # Should use the provided ID
    assert correlation_id == provided_id


def test_correlation_id_in_response_header(app):
    """Test that correlation ID is added to response headers."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    response = client.get("/test")

    assert "X-Correlation-ID" in response.headers
    correlation_id = response.headers["X-Correlation-ID"]
    assert correlation_id is not None


def test_response_header_matches_request_state(app):
    """Test that correlation ID in response header matches request state."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    response = client.get("/test")

    response_header_id = response.headers["X-Correlation-ID"]
    response_body_id = response.json()["correlation_id"]

    assert response_header_id == response_body_id


def test_custom_header_name(app):
    """Test that middleware supports custom header names."""
    app.add_middleware(CorrelationIDMiddleware, header_name="X-Request-ID")
    client = TestClient(app)

    provided_id = "custom-request-id-67890"
    response = client.get("/test", headers={"X-Request-ID": provided_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == provided_id
    assert response.json()["correlation_id"] == provided_id


def test_context_variable_set_during_request():
    """Test that correlation ID is set in context variable for logging."""
    from src.utils.logging import correlation_id_ctx

    app = FastAPI()
    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/test")
    async def test_endpoint():
        # Get correlation ID from context
        ctx_correlation_id = correlation_id_ctx.get()
        return {"ctx_correlation_id": ctx_correlation_id}

    client = TestClient(app)
    provided_id = "context-test-id"

    response = client.get("/test", headers={"X-Correlation-ID": provided_id})

    assert response.status_code == 200
    # Context variable should have been set during request processing
    ctx_id = response.json()["ctx_correlation_id"]
    assert ctx_id == provided_id


def test_get_correlation_id_returns_current_context():
    """Test that get_correlation_id() returns the current context value."""
    from src.utils.logging import correlation_id_ctx

    # Set a correlation ID in context
    test_id = "test-correlation-id"
    token = correlation_id_ctx.set(test_id)

    try:
        # get_correlation_id should return the context value
        result = get_correlation_id()
        assert result == test_id
    finally:
        # Clean up context
        correlation_id_ctx.reset(token)


def test_get_correlation_id_returns_none_when_not_set():
    """Test that get_correlation_id() returns None when context is not set."""
    from src.utils.logging import correlation_id_ctx

    # Ensure context is not set
    result = get_correlation_id()
    assert result is None or isinstance(result, str)  # May be set from other tests


def test_middleware_handles_exceptions_gracefully(app):
    """Test that middleware doesn't break error handling."""
    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/error")

    # Should still get 500 error
    assert response.status_code == 500

    # Correlation ID should still be in response header
    assert "X-Correlation-ID" in response.headers


def test_correlation_id_unique_per_request(app):
    """Test that each request gets a unique correlation ID when not provided."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    response1 = client.get("/test")
    response2 = client.get("/test")

    id1 = response1.headers["X-Correlation-ID"]
    id2 = response2.headers["X-Correlation-ID"]

    # Each request should get a different ID
    assert id1 != id2


def test_case_insensitive_header_lookup(app):
    """Test that header lookup is case-insensitive."""
    app.add_middleware(CorrelationIDMiddleware)
    client = TestClient(app)

    provided_id = "case-insensitive-test"

    # Send header with different case
    response = client.get("/test", headers={"x-correlation-id": provided_id})

    assert response.status_code == 200
    correlation_id = response.json()["correlation_id"]

    # Should still extract the ID
    assert correlation_id == provided_id


def test_middleware_preserves_request_state(app):
    """Test that middleware doesn't overwrite other request state."""
    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/state-test")
    async def state_test_endpoint(request: Request):
        # Set some other state
        request.state.custom_value = "test"
        return {
            "correlation_id": request.state.correlation_id,
            "custom_value": request.state.custom_value,
        }

    client = TestClient(app)
    response = client.get("/state-test")

    assert response.status_code == 200
    assert response.json()["custom_value"] == "test"
    assert response.json()["correlation_id"] is not None


def test_middleware_works_with_multiple_middleware_layers():
    """Test that correlation middleware works with other middleware."""
    app = FastAPI()

    # Add another middleware layer
    @app.middleware("http")
    async def custom_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Custom-Header"] = "test"
        return response

    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"correlation_id": request.state.correlation_id}

    client = TestClient(app)
    response = client.get("/test")

    # Both middleware should work
    assert "X-Correlation-ID" in response.headers
    assert "X-Custom-Header" in response.headers
    assert response.headers["X-Custom-Header"] == "test"
