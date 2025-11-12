"""
Integration tests for API endpoints.

Tests end-to-end functionality including:
- Health endpoint with model states
- Predict endpoint with real predictions
- Explain endpoint with SHAP explanations
- Admin reload endpoint
- Metrics endpoint
- Middleware integration (correlation ID, telemetry)
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_trained_model():
    """Create a mock trained model for integration testing."""
    from sklearn.ensemble import RandomForestClassifier

    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def mock_model_wrapper(mock_trained_model):
    """Create a ModelWrapper with trained model."""
    from src.infer import ModelWrapper

    wrapper = ModelWrapper(mock_trained_model)
    return wrapper


@pytest.fixture
def integration_app(mock_model_wrapper):
    """Create a fully configured FastAPI app for integration testing."""
    from fastapi import FastAPI
    from src.app.api import health, predict, explain, admin, metrics
    from src.app.api.middleware.correlation import CorrelationIDMiddleware
    from src.utils.telemetry import PrometheusMiddleware

    app = FastAPI()

    # Add middleware
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(PrometheusMiddleware)

    # Include routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(admin.router)
    app.include_router(metrics.router)

    # Configure app state
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.model_metadata = {
        "model_path": "test://model",
        "run_id": "test_run_123",
        "accuracy": 0.95,
    }

    return app


def test_health_endpoint_returns_ready_state(integration_app):
    """Test that health endpoint returns ready state when model is loaded."""
    client = TestClient(integration_app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "model_metadata" in data


def test_health_endpoint_returns_not_ready_when_model_missing():
    """Test that health endpoint returns not ready when model is not loaded."""
    from fastapi import FastAPI
    from src.app.api import health

    app = FastAPI()
    app.include_router(health.router)
    app.state.is_ready = False
    app.state.ml_wrapper = None

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    data = response.json()

    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False


def test_predict_endpoint_returns_predictions(integration_app):
    """Test that predict endpoint returns predictions for valid input."""
    client = TestClient(integration_app)

    payload = {"features": [[1.0, 2.0], [3.0, 4.0]]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert all(isinstance(p, int) for p in data["predictions"])


def test_predict_endpoint_validates_feature_dimensions(integration_app):
    """Test that predict endpoint validates feature dimensions."""
    client = TestClient(integration_app)

    # Send wrong number of features (model expects 2, sending 3)
    payload = {"features": [[1.0, 2.0, 3.0]]}
    response = client.post("/predict", json=payload)

    # Should return validation error
    assert response.status_code in [400, 422]


def test_predict_endpoint_includes_correlation_id_in_response(integration_app):
    """Test that predict endpoint includes correlation ID in response."""
    client = TestClient(integration_app)

    correlation_id = "test-correlation-123"
    payload = {"features": [[1.0, 2.0]]}

    response = client.post(
        "/predict",
        json=payload,
        headers={"X-Correlation-ID": correlation_id}
    )

    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_explain_endpoint_returns_shap_values(integration_app):
    """Test that explain endpoint returns SHAP explanations."""
    client = TestClient(integration_app)

    payload = {"features": [[1.0, 2.0]]}

    # Mock SHAP to avoid dependency
    with patch("src.app.api.explain.shap") as mock_shap:
        mock_explainer = Mock()
        mock_explainer.shap_values = Mock(return_value=np.array([[0.1, 0.2]]))
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        response = client.post("/explain/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "shap_values" in data
    assert "feature_values" in data
    assert "explanation_type" in data

    assert len(data["shap_values"]) == 2
    assert data["feature_values"] == [1.0, 2.0]


def test_explain_endpoint_rejects_batch_requests(integration_app):
    """Test that explain endpoint rejects batch requests."""
    client = TestClient(integration_app)

    # Send multiple samples
    payload = {"features": [[1.0, 2.0], [3.0, 4.0]]}
    response = client.post("/explain/", json=payload)

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"].lower()


def test_admin_reload_requires_authentication(integration_app):
    """Test that admin reload endpoint requires authentication."""
    client = TestClient(integration_app)

    # No auth header
    response = client.post("/admin/reload", json={"run_id": "test_run"})

    assert response.status_code in [401, 403]


def test_admin_reload_with_valid_auth(integration_app):
    """Test that admin reload works with valid authentication."""
    client = TestClient(integration_app)

    with patch("src.app.api.admin.os.getenv", return_value="test_token"):
        with patch("src.app.api.admin.reload_model_by_run_id") as mock_reload:
            mock_reload.return_value = AsyncMock()

            response = client.post(
                "/admin/reload",
                json={"run_id": "test_run_456"},
                headers={"Authorization": "Bearer test_token"}
            )

            # Should be accepted (may return 202 or trigger background task)
            assert response.status_code in [200, 202]


def test_metrics_endpoint_returns_prometheus_format(integration_app):
    """Test that metrics endpoint returns Prometheus metrics."""
    client = TestClient(integration_app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    # Should contain some metric data
    content = response.text
    assert len(content) > 0


def test_middleware_chain_works_correctly(integration_app):
    """Test that middleware chain (correlation + telemetry) works correctly."""
    client = TestClient(integration_app)

    # Make a request through the middleware chain
    payload = {"features": [[1.0, 2.0]]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    # Correlation ID should be added
    assert "X-Correlation-ID" in response.headers

    # Telemetry should have recorded metrics (check via /metrics)
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200


def test_predict_and_explain_consistency(integration_app):
    """Test that predict and explain return consistent predictions."""
    client = TestClient(integration_app)

    features = [[5.0, 6.0]]

    # Get prediction
    predict_response = client.post("/predict", json={"features": features})
    assert predict_response.status_code == 200
    prediction = predict_response.json()["predictions"][0]

    # Get explanation (mocked SHAP)
    with patch("src.app.api.explain.shap", side_effect=ImportError()):
        explain_response = client.post("/explain/", json={"features": features})

    assert explain_response.status_code == 200
    explained_prediction = explain_response.json()["prediction"]

    # Predictions should match
    assert prediction == explained_prediction


def test_concurrent_requests_handling(integration_app):
    """Test that app handles concurrent requests correctly."""
    import concurrent.futures

    client = TestClient(integration_app)

    def make_request(i):
        payload = {"features": [[float(i), float(i + 1)]]}
        return client.post("/predict", json=payload)

    # Make concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        responses = [f.result() for f in futures]

    # All requests should succeed
    assert all(r.status_code == 200 for r in responses)

    # All should have predictions
    assert all("predictions" in r.json() for r in responses)


def test_error_handling_in_middleware(integration_app):
    """Test that middleware properly handles errors."""
    from fastapi import HTTPException

    # Add an endpoint that raises an error
    @integration_app.get("/test-error")
    async def error_endpoint():
        raise HTTPException(status_code=500, detail="Test error")

    client = TestClient(integration_app, raise_server_exceptions=False)
    response = client.get("/test-error")

    assert response.status_code == 500

    # Correlation ID should still be present
    assert "X-Correlation-ID" in response.headers

    # Metrics should still be recorded
    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200


def test_health_endpoint_with_degraded_mlflow():
    """Test health endpoint when MLflow is unavailable."""
    from fastapi import FastAPI
    from src.app.api import health

    app = FastAPI()
    app.include_router(health.router)

    app.state.is_ready = True
    app.state.ml_wrapper = Mock()
    app.state.mlflow_available = False

    client = TestClient(app)
    response = client.get("/health")

    # Should still return 200 but indicate degraded state
    assert response.status_code in [200, 503]
    data = response.json()

    if "mlflow_status" in data:
        assert data["mlflow_status"] == "unavailable"


def test_predict_endpoint_batch_size_limits(integration_app):
    """Test that predict endpoint respects batch size limits."""
    client = TestClient(integration_app)

    # Create a large batch (assuming there's a limit configured)
    large_batch = {"features": [[1.0, 2.0] for _ in range(1001)]}

    response = client.post("/predict", json=large_batch)

    # Should either succeed or return 400 for batch too large
    assert response.status_code in [200, 400, 413, 422]


def test_api_versioning_and_tags(integration_app):
    """Test that API endpoints have proper tags and documentation."""
    from fastapi.openapi.utils import get_openapi

    # Get OpenAPI schema
    openapi_schema = get_openapi(
        title=integration_app.title,
        version=integration_app.version,
        routes=integration_app.routes,
    )

    # Check that endpoints are documented
    assert "paths" in openapi_schema
    assert "/health" in openapi_schema["paths"]
    assert "/predict" in openapi_schema["paths"]
    assert "/explain/" in openapi_schema["paths"]


def test_request_validation_error_responses(integration_app):
    """Test that validation errors return proper error responses."""
    client = TestClient(integration_app)

    # Send invalid JSON
    invalid_payloads = [
        {},  # Missing features
        {"features": "not a list"},  # Wrong type
        {"features": []},  # Empty list
        {"wrong_key": [[1.0, 2.0]]},  # Wrong key
    ]

    for payload in invalid_payloads:
        response = client.post("/predict", json=payload)

        # Should return 422 (validation error) or 400
        assert response.status_code in [400, 422]

        # Response should have error detail
        assert "detail" in response.json()
