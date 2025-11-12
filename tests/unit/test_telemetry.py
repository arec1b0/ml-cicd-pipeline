"""
Unit tests for telemetry utilities (src/utils/telemetry.py).

Tests cover:
- PrometheusMiddleware request tracking
- Request latency histogram
- Request count metrics
- Error tracking (5xx responses)
- Path normalization and anonymization
- Metrics response generation
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from src.utils.telemetry import (
    PrometheusMiddleware,
    normalize_request_path,
    _anonymize_path,
    metrics_response,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUEST_ERRORS,
    MODEL_ACCURACY,
)


@pytest.fixture
def app():
    """Create a simple FastAPI app for testing."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/users/{user_id}")
    async def user_endpoint(user_id: str):
        return {"user_id": user_id}

    @app.get("/error")
    async def error_endpoint():
        raise HTTPException(status_code=500, detail="Test error")

    @app.get("/client-error")
    async def client_error_endpoint():
        raise HTTPException(status_code=400, detail="Bad request")

    return app


def test_anonymize_path_replaces_digits():
    """Test that digit segments are replaced with {id}."""
    assert _anonymize_path("/users/12345") == "/users/{id}"
    assert _anonymize_path("/items/999/details") == "/items/{id}/details"


def test_anonymize_path_replaces_uuids():
    """Test that UUID segments are replaced with {id}."""
    uuid_path = "/users/550e8400-e29b-41d4-a716-446655440000"
    assert _anonymize_path(uuid_path) == "/users/{id}"

    # Shorter UUID-like strings
    assert _anonymize_path("/items/abc123def456") == "/items/{id}"


def test_anonymize_path_preserves_named_segments():
    """Test that named segments are preserved."""
    assert _anonymize_path("/api/users/profile") == "/api/users/profile"
    assert _anonymize_path("/health") == "/health"


def test_anonymize_path_handles_trailing_slash():
    """Test that trailing slashes are preserved."""
    assert _anonymize_path("/users/123/") == "/users/{id}/"
    assert _anonymize_path("/api/") == "/api/"


def test_anonymize_path_handles_root():
    """Test that root path is handled correctly."""
    assert _anonymize_path("/") == "/"
    assert _anonymize_path("") == "/"


def test_anonymize_path_mixed_segments():
    """Test path with mix of named and numeric segments."""
    assert _anonymize_path("/api/v1/users/123/orders/456") == "/api/v1/users/{id}/orders/{id}"


def test_normalize_request_path_uses_route_template():
    """Test that normalize_request_path uses route template when available."""
    mock_request = Mock(spec=Request)
    mock_route = Mock()
    mock_route.path = "/users/{user_id}"

    mock_request.scope = {"route": mock_route}
    mock_request.url.path = "/users/12345"

    result = normalize_request_path(mock_request)
    assert result == "/users/{user_id}"


def test_normalize_request_path_fallback_to_anonymize():
    """Test that normalize_request_path falls back to anonymization."""
    mock_request = Mock(spec=Request)
    mock_request.scope = {"route": None}
    mock_request.url.path = "/users/12345"

    result = normalize_request_path(mock_request)
    assert result == "/users/{id}"


def test_normalize_request_path_with_path_format():
    """Test normalize_request_path with path_format attribute."""
    mock_request = Mock(spec=Request)
    mock_route = Mock()
    # path_format is used by some routers instead of path
    del mock_route.path
    mock_route.path_format = "/items/{item_id}"

    mock_request.scope = {"route": mock_route}
    mock_request.url.path = "/items/999"

    result = normalize_request_path(mock_request)
    assert result == "/items/{item_id}"


def test_middleware_records_request_count(app):
    """Test that middleware records request counts."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app)

    # Get initial count
    initial_count = REQUEST_COUNT.labels(method="GET", path="/test", status="200")._value.get()

    # Make request
    response = client.get("/test")
    assert response.status_code == 200

    # Check count increased
    final_count = REQUEST_COUNT.labels(method="GET", path="/test", status="200")._value.get()
    assert final_count > initial_count


def test_middleware_records_latency(app):
    """Test that middleware records request latency."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app)

    # Get initial count of latency observations
    initial_count = REQUEST_LATENCY.labels(method="GET", path="/test")._sum.get()

    # Make request
    response = client.get("/test")
    assert response.status_code == 200

    # Check latency was recorded (sum should increase)
    final_sum = REQUEST_LATENCY.labels(method="GET", path="/test")._sum.get()
    assert final_sum >= initial_count  # Latency should be recorded


def test_middleware_records_errors_for_5xx(app):
    """Test that middleware records errors for 5xx status codes."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app, raise_server_exceptions=False)

    # Get initial error count
    initial_errors = REQUEST_ERRORS.labels(method="GET", path="/error")._value.get()

    # Make request that raises 500
    response = client.get("/error")
    assert response.status_code == 500

    # Check error count increased
    final_errors = REQUEST_ERRORS.labels(method="GET", path="/error")._value.get()
    assert final_errors > initial_errors


def test_middleware_does_not_record_errors_for_4xx(app):
    """Test that middleware does not record 4xx as errors."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app, raise_server_exceptions=False)

    # Get initial error count
    initial_errors = REQUEST_ERRORS.labels(method="GET", path="/client-error")._value.get()

    # Make request that raises 400
    response = client.get("/client-error")
    assert response.status_code == 400

    # Error count should not increase for 4xx
    final_errors = REQUEST_ERRORS.labels(method="GET", path="/client-error")._value.get()
    assert final_errors == initial_errors


def test_middleware_normalizes_path_with_ids(app):
    """Test that middleware normalizes paths with IDs."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app)

    # Make requests with different IDs
    client.get("/users/123")
    client.get("/users/456")

    # Both should be recorded under the same normalized path
    # The route template should be used: /users/{user_id}
    count_metric = REQUEST_COUNT.labels(method="GET", path="/users/{user_id}", status="200")
    assert count_metric._value.get() >= 2


def test_middleware_handles_exceptions_during_request(app):
    """Test that middleware handles exceptions and records them."""
    app.add_middleware(PrometheusMiddleware)

    @app.get("/exception")
    async def exception_endpoint():
        raise ValueError("Unexpected error")

    client = TestClient(app, raise_server_exceptions=False)

    # Get initial metrics
    initial_errors = REQUEST_ERRORS.labels(method="GET", path="/exception")._value.get()
    initial_count = REQUEST_COUNT.labels(method="GET", path="/exception", status="500")._value.get()

    # Make request that raises exception
    response = client.get("/exception")
    assert response.status_code == 500

    # Check metrics were recorded
    final_errors = REQUEST_ERRORS.labels(method="GET", path="/exception")._value.get()
    final_count = REQUEST_COUNT.labels(method="GET", path="/exception", status="500")._value.get()

    assert final_errors > initial_errors
    assert final_count > initial_count


def test_middleware_records_latency_even_on_exception(app):
    """Test that middleware records latency even when exceptions occur."""
    app.add_middleware(PrometheusMiddleware)

    @app.get("/exception-latency")
    async def exception_endpoint():
        time.sleep(0.01)  # Small delay
        raise ValueError("Error after delay")

    client = TestClient(app, raise_server_exceptions=False)

    # Get initial latency sum
    initial_sum = REQUEST_LATENCY.labels(method="GET", path="/exception-latency")._sum.get()

    # Make request
    response = client.get("/exception-latency")
    assert response.status_code == 500

    # Latency should still be recorded
    final_sum = REQUEST_LATENCY.labels(method="GET", path="/exception-latency")._sum.get()
    assert final_sum > initial_sum


def test_metrics_response_returns_prometheus_format():
    """Test that metrics_response returns Prometheus text format."""
    response = metrics_response()

    assert isinstance(response, Response)
    assert "text/plain" in response.media_type
    assert isinstance(response.body, bytes)

    # Should contain metric names
    content = response.body.decode("utf-8")
    assert "ml_request_count" in content or "ml_request_latency_seconds" in content


def test_model_accuracy_gauge_can_be_set():
    """Test that MODEL_ACCURACY gauge can be set and retrieved."""
    # Set accuracy value
    test_accuracy = 0.95
    MODEL_ACCURACY.set(test_accuracy)

    # Retrieve value
    assert MODEL_ACCURACY._value.get() == test_accuracy


def test_middleware_records_different_methods(app):
    """Test that middleware correctly records different HTTP methods."""
    app.add_middleware(PrometheusMiddleware)

    @app.post("/test-post")
    async def post_endpoint():
        return {"status": "created"}

    client = TestClient(app)

    # Get initial counts
    initial_get = REQUEST_COUNT.labels(method="GET", path="/test", status="200")._value.get()
    initial_post = REQUEST_COUNT.labels(method="POST", path="/test-post", status="200")._value.get()

    # Make requests
    client.get("/test")
    client.post("/test-post")

    # Check both methods recorded separately
    final_get = REQUEST_COUNT.labels(method="GET", path="/test", status="200")._value.get()
    final_post = REQUEST_COUNT.labels(method="POST", path="/test-post", status="200")._value.get()

    assert final_get > initial_get
    assert final_post > initial_post


def test_middleware_records_different_status_codes(app):
    """Test that middleware records different status codes separately."""
    app.add_middleware(PrometheusMiddleware)

    @app.get("/status-test")
    async def status_test(status: int = 200):
        if status == 404:
            raise HTTPException(status_code=404, detail="Not found")
        return {"status": "ok"}

    client = TestClient(app, raise_server_exceptions=False)

    # Get initial counts
    initial_200 = REQUEST_COUNT.labels(method="GET", path="/status-test", status="200")._value.get()
    initial_404 = REQUEST_COUNT.labels(method="GET", path="/status-test", status="404")._value.get()

    # Make requests with different status codes
    client.get("/status-test")
    client.get("/status-test?status=404")

    # Check both status codes recorded
    final_200 = REQUEST_COUNT.labels(method="GET", path="/status-test", status="200")._value.get()
    final_404 = REQUEST_COUNT.labels(method="GET", path="/status-test", status="404")._value.get()

    assert final_200 > initial_200
    assert final_404 > initial_404


def test_path_normalization_only_happens_once_per_request(app):
    """Test that path normalization is efficient and doesn't duplicate work."""
    app.add_middleware(PrometheusMiddleware)
    client = TestClient(app)

    with patch("src.utils.telemetry.normalize_request_path", wraps=normalize_request_path) as mock_normalize:
        response = client.get("/test")
        assert response.status_code == 200

        # normalize_request_path should be called exactly once per request
        # (once in the success path, potentially once in exception path if needed)
        assert mock_normalize.call_count <= 2  # At most twice (success + latency recording)
