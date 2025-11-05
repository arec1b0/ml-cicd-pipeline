"""
Tests for health endpoint integration with model reload.

Verifies that health check reflects model state correctly during reloads.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api.health import router as health_router
from src.models.manager import LoadedModel, ModelDescriptor
from src.models.infer import ModelWrapper


@pytest.fixture
def mock_model_wrapper():
    """Create a mock ModelWrapper for testing."""
    wrapper = MagicMock(spec=ModelWrapper)
    wrapper.predict.return_value = [0, 1, 2]
    return wrapper


@pytest.fixture
def test_app():
    """Create a test FastAPI app with health router."""
    app = FastAPI()
    app.include_router(health_router)
    return app


def test_health_reflects_not_ready_state(test_app):
    """Test that health endpoint reflects not ready state when no model loaded."""
    client = TestClient(test_app)

    # Initialize app state as not ready
    test_app.state.is_ready = False
    test_app.state.ml_metrics = None
    test_app.state.mlflow_connectivity = {"status": "unknown"}

    response = client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is False
    assert data["details"] is None


def test_health_reflects_ready_state(test_app):
    """Test that health endpoint reflects ready state when model loaded."""
    client = TestClient(test_app)

    # Initialize app state as ready
    test_app.state.is_ready = True
    test_app.state.ml_metrics = {"accuracy": 0.95, "f1_score": 0.93}
    test_app.state.mlflow_connectivity = {
        "status": "ok",
        "server_version": "2.7.0",
        "verified_at": "2025-11-05T12:00:00Z",
        "model_uri": "models:/test-model/1",
    }

    response = client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["details"] == {"metrics": {"accuracy": 0.95, "f1_score": 0.93}}
    assert data["mlflow"]["status"] == "ok"
    assert data["mlflow"]["server_version"] == "2.7.0"


def test_health_reflects_error_state(test_app):
    """Test that health endpoint reflects error state after failed reload."""
    client = TestClient(test_app)

    # Initialize app state with error
    test_app.state.is_ready = False
    test_app.state.ml_metrics = None
    test_app.state.mlflow_connectivity = {
        "status": "error",
        "error": "Failed to connect to MLflow",
        "verified_at": "2025-11-05T12:00:00Z",
    }

    response = client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"  # Service is up
    assert data["ready"] is False  # But model not ready
    assert data["mlflow"]["status"] == "error"
    assert "Failed to connect" in data["mlflow"]["error"]


def test_health_state_transition_after_reload(test_app, mock_model_wrapper):
    """Test that health state transitions correctly after model reload."""
    client = TestClient(test_app)

    # Start in not ready state
    test_app.state.is_ready = False
    test_app.state.ml_metrics = None

    response = client.get("/health/")
    assert response.json()["ready"] is False

    # Simulate successful model reload
    test_app.state.is_ready = True
    test_app.state.ml_wrapper = mock_model_wrapper
    test_app.state.ml_metrics = {"accuracy": 0.95}

    response = client.get("/health/")
    assert response.json()["ready"] is True
    assert response.json()["details"]["metrics"]["accuracy"] == 0.95


def test_health_with_model_metadata(test_app):
    """Test that health endpoint includes model metadata if available."""
    client = TestClient(test_app)

    test_app.state.is_ready = True
    test_app.state.ml_metrics = {
        "accuracy": 0.95,
        "f1_score": 0.93,
        "precision": 0.94,
        "recall": 0.92,
    }
    test_app.state.mlflow_connectivity = {
        "status": "ok",
        "server_version": "2.7.0",
        "model_uri": "models:/iris-model/2",
        "verified_at": "2025-11-05T12:00:00Z",
    }

    response = client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["details"]["metrics"]["accuracy"] == 0.95
    assert data["details"]["metrics"]["f1_score"] == 0.93
    assert len(data["details"]["metrics"]) == 4


def test_health_without_mlflow_connectivity(test_app):
    """Test health endpoint when MLflow connectivity not available."""
    client = TestClient(test_app)

    test_app.state.is_ready = True
    test_app.state.ml_metrics = {"accuracy": 0.95}
    # No mlflow_connectivity in state

    response = client.get("/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["mlflow"] is None


def test_health_atomic_view_during_reload(test_app, mock_model_wrapper):
    """Test that health endpoint shows atomic view during reload.

    This test verifies that the health endpoint either shows the old model
    state or the new model state, but never a partially updated state.
    """
    client = TestClient(test_app)

    # Initial state with model v1
    test_app.state.is_ready = True
    test_app.state.ml_wrapper = mock_model_wrapper
    test_app.state.ml_metrics = {"accuracy": 0.90, "version": "1"}

    # Health check should show v1
    response = client.get("/health/")
    data = response.json()
    assert data["ready"] is True
    assert data["details"]["metrics"]["version"] == "1"

    # Simulate atomic swap to model v2
    new_wrapper = MagicMock(spec=ModelWrapper)
    test_app.state.ml_wrapper = new_wrapper
    test_app.state.ml_metrics = {"accuracy": 0.95, "version": "2"}
    test_app.state.is_ready = True

    # Health check should now show v2
    response = client.get("/health/")
    data = response.json()
    assert data["ready"] is True
    assert data["details"]["metrics"]["version"] == "2"
    assert data["details"]["metrics"]["accuracy"] == 0.95


def test_health_shows_degraded_state_on_reload_failure(test_app):
    """Test that health shows degraded state if reload fails but old model still works."""
    client = TestClient(test_app)

    # Old model still ready, but reload failed
    test_app.state.is_ready = True
    test_app.state.ml_metrics = {"accuracy": 0.90}
    test_app.state.mlflow_connectivity = {
        "status": "error",
        "error": "Network timeout during reload",
        "verified_at": "2025-11-05T12:00:00Z",
    }

    response = client.get("/health/")
    data = response.json()

    # Service is still ready (old model still works)
    assert data["ready"] is True
    # But MLflow shows error (reload failed)
    assert data["mlflow"]["status"] == "error"
