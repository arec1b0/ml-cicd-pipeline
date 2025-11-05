"""
Tests for admin reload endpoint.

Verifies authentication, atomic model swapping, and error handling.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api.admin import router as admin_router
from src.models.manager import LoadedModel, ModelDescriptor, ModelManager
from src.models.infer import ModelWrapper


@pytest.fixture
def mock_model_wrapper():
    """Create a mock ModelWrapper for testing."""
    wrapper = MagicMock(spec=ModelWrapper)
    wrapper.predict.return_value = [0, 1, 2]
    return wrapper


@pytest.fixture
def mock_loaded_model(mock_model_wrapper, tmp_path):
    """Create a mock LoadedModel for testing."""
    descriptor = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test-model/1",
        version="1",
        stage="Production",
        run_id="test-run-123",
        server_version="2.7.0",
    )

    return LoadedModel(
        wrapper=mock_model_wrapper,
        model_file=tmp_path / "model.pkl",
        metrics={"accuracy": 0.95, "f1_score": 0.93},
        accuracy=0.95,
        descriptor=descriptor,
        artifact_path=tmp_path,
    )


@pytest.fixture
def test_app(mock_loaded_model):
    """Create a test FastAPI app with admin router."""
    app = FastAPI()
    app.include_router(admin_router)

    # Initialize app state
    app.state.admin_api_token = "test-secret-token"
    app.state.admin_token_header = "X-Admin-Token"
    app.state.is_ready = True
    app.state.ml_wrapper = mock_loaded_model.wrapper
    app.state.ml_metrics = mock_loaded_model.metrics

    # Mock model manager
    mock_manager = MagicMock(spec=ModelManager)
    app.state.model_manager = mock_manager

    # Mock apply function
    def mock_apply_fn(state: LoadedModel):
        app.state.ml_wrapper = state.wrapper
        app.state.ml_metrics = state.metrics
        app.state.is_ready = True

    app.state.apply_model_state = mock_apply_fn

    return app


def test_reload_endpoint_requires_authentication(test_app):
    """Test that reload endpoint requires valid authentication token."""
    client = TestClient(test_app)

    # Test without token
    response = client.post("/admin/reload")
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"

    # Test with invalid token
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "wrong-token"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


def test_reload_endpoint_with_valid_token(test_app, mock_loaded_model):
    """Test successful model reload with valid authentication."""
    client = TestClient(test_app)

    # Mock the manager.reload to return a new model state
    test_app.state.model_manager.reload = AsyncMock(return_value=mock_loaded_model)

    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reloaded"
    assert data["detail"] == "Model reloaded successfully"
    assert data["version"] == "1"
    assert data["stage"] == "Production"

    # Verify manager.reload was called with force=True
    test_app.state.model_manager.reload.assert_awaited_once_with(force=True)


def test_reload_endpoint_noop_when_unchanged(test_app):
    """Test that endpoint returns noop when model descriptor unchanged."""
    client = TestClient(test_app)

    # Mock the manager.reload to return None (no change)
    test_app.state.model_manager.reload = AsyncMock(return_value=None)

    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "noop"
    assert data["detail"] == "Model descriptor unchanged"


def test_reload_endpoint_handles_errors(test_app):
    """Test that reload endpoint handles errors gracefully."""
    client = TestClient(test_app)

    # Mock the manager.reload to raise an exception
    test_app.state.model_manager.reload = AsyncMock(
        side_effect=Exception("Failed to download model")
    )

    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 500
    assert "Reload failed" in response.json()["detail"]
    assert "Failed to download model" in response.json()["detail"]


def test_reload_endpoint_manager_unavailable(test_app):
    """Test that endpoint returns 503 when manager unavailable."""
    client = TestClient(test_app)

    # Remove model manager from app state
    test_app.state.model_manager = None

    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model manager unavailable"


def test_reload_endpoint_apply_function_unavailable(test_app):
    """Test that endpoint returns 503 when apply function unavailable."""
    client = TestClient(test_app)

    # Remove apply function from app state
    test_app.state.apply_model_state = None

    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model manager unavailable"


def test_reload_endpoint_without_auth_configured(mock_loaded_model):
    """Test that endpoint works without authentication when not configured."""
    app = FastAPI()
    app.include_router(admin_router)

    # Initialize app state without auth token
    app.state.admin_api_token = None
    app.state.admin_token_header = "X-Admin-Token"

    # Mock model manager
    mock_manager = MagicMock(spec=ModelManager)
    mock_manager.reload = AsyncMock(return_value=mock_loaded_model)
    app.state.model_manager = mock_manager

    def mock_apply_fn(state: LoadedModel):
        pass

    app.state.apply_model_state = mock_apply_fn

    client = TestClient(app)

    # Should succeed without token
    response = client.post("/admin/reload")
    assert response.status_code == 200


def test_atomic_swap_on_successful_reload(test_app, mock_loaded_model):
    """Test that model swap is atomic - old model stays until new one loads."""
    client = TestClient(test_app)

    # Store original wrapper
    original_wrapper = test_app.state.ml_wrapper

    # Create a new wrapper for the reloaded model
    new_wrapper = MagicMock(spec=ModelWrapper)
    new_wrapper.predict.return_value = [3, 4, 5]

    new_model = LoadedModel(
        wrapper=new_wrapper,
        model_file=Path("/tmp/new_model.pkl"),
        metrics={"accuracy": 0.97},
        accuracy=0.97,
        descriptor=ModelDescriptor(
            source="mlflow",
            model_uri="models:/test-model/2",
            version="2",
            stage="Production",
        ),
        artifact_path=Path("/tmp"),
    )

    # Mock the reload to return new model
    test_app.state.model_manager.reload = AsyncMock(return_value=new_model)

    # Before reload, should have original wrapper
    assert test_app.state.ml_wrapper is original_wrapper

    # Perform reload
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 200

    # After reload, should have new wrapper
    assert test_app.state.ml_wrapper is new_wrapper
    assert test_app.state.ml_wrapper is not original_wrapper


def test_atomic_swap_on_failed_reload(test_app):
    """Test that old model stays active if reload fails."""
    client = TestClient(test_app)

    # Store original wrapper
    original_wrapper = test_app.state.ml_wrapper

    # Mock the reload to fail
    test_app.state.model_manager.reload = AsyncMock(
        side_effect=Exception("Network error")
    )

    # Before reload, should have original wrapper
    assert test_app.state.ml_wrapper is original_wrapper

    # Attempt reload (should fail)
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret-token"}
    )

    assert response.status_code == 500

    # After failed reload, should still have original wrapper
    assert test_app.state.ml_wrapper is original_wrapper


def test_custom_auth_header_name(mock_loaded_model):
    """Test that custom authentication header names work."""
    app = FastAPI()
    app.include_router(admin_router)

    # Use custom header name
    app.state.admin_api_token = "custom-token"
    app.state.admin_token_header = "X-Custom-Auth-Header"

    # Mock model manager
    mock_manager = MagicMock(spec=ModelManager)
    mock_manager.reload = AsyncMock(return_value=mock_loaded_model)
    app.state.model_manager = mock_manager

    def mock_apply_fn(state: LoadedModel):
        pass

    app.state.apply_model_state = mock_apply_fn

    client = TestClient(app)

    # Should fail with wrong header
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "custom-token"}
    )
    assert response.status_code == 403

    # Should succeed with correct custom header
    response = client.post(
        "/admin/reload",
        headers={"X-Custom-Auth-Header": "custom-token"}
    )
    assert response.status_code == 200
