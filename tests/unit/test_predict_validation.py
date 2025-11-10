"""
Tests for prediction endpoint feature dimension validation.

Verifies that the predict endpoint correctly validates input feature dimensions
against the expected dimension derived from the model.
"""

from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api.predict import router as predict_router
from src.models.infer import ModelWrapper


@pytest.fixture
def mock_model_wrapper():
    """Create a mock ModelWrapper for testing."""
    wrapper = MagicMock(spec=ModelWrapper)
    wrapper.predict.return_value = [0, 1, 2]
    wrapper.get_input_dimension.return_value = 4  # Iris dataset has 4 features
    return wrapper


@pytest.fixture
def test_app(mock_model_wrapper):
    """Create a test FastAPI app with predict router and mocked model."""
    app = FastAPI()
    app.include_router(predict_router)

    # Initialize app state with loaded model
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.expected_feature_dimension = 4  # Set from model at startup
    app.state.model_metadata = {
        "source": "local",
        "model_uri": "test://model",
        "expected_feature_dimension": 4,
    }

    return app


def test_predict_with_correct_dimension(test_app):
    """Test that prediction succeeds with correct feature dimension."""
    client = TestClient(test_app)

    # Valid request with 4 features (Iris dataset)
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_with_incorrect_dimension_too_many(test_app):
    """Test that prediction fails with too many features."""
    client = TestClient(test_app)

    # Invalid request with 10 features instead of 4
    payload = {
        "features": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "Invalid feature dimension" in data["detail"]
    assert "Expected 4, got 10" in data["detail"]


def test_predict_with_incorrect_dimension_too_few(test_app):
    """Test that prediction fails with too few features."""
    client = TestClient(test_app)

    # Invalid request with 2 features instead of 4
    payload = {
        "features": [
            [1.0, 2.0],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "Invalid feature dimension" in data["detail"]
    assert "Expected 4, got 2" in data["detail"]


def test_predict_dimension_validation_per_sample(test_app):
    """Test that dimension validation is performed for each sample."""
    client = TestClient(test_app)

    # Mixed request: first sample valid, second sample invalid
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],  # Valid: 4 features
            [1.0, 2.0, 3.0],        # Invalid: 3 features
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "Invalid feature dimension at index 1" in data["detail"]
    assert "Expected 4, got 3" in data["detail"]


def test_predict_when_model_not_ready(test_app):
    """Test that prediction fails when model is not ready."""
    client = TestClient(test_app)

    # Set model as not ready
    test_app.state.is_ready = False

    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 503
    data = response.json()
    assert "Model not loaded" in data["detail"]


def test_predict_with_empty_features(test_app):
    """Test that prediction fails with empty feature list."""
    client = TestClient(test_app)

    payload = {
        "features": []
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "cannot be empty" in data["detail"]


def test_predict_fallback_to_default_dimension(mock_model_wrapper):
    """Test that prediction falls back to default dimension if not set in app state."""
    app = FastAPI()
    app.include_router(predict_router)

    # Initialize app state without expected_feature_dimension
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    # Note: expected_feature_dimension is not set, should fall back to default (4)

    client = TestClient(app)

    # Valid request with 4 features (default dimension)
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 200


def test_predict_uses_model_derived_dimension(mock_model_wrapper):
    """Test that prediction uses model-derived dimension when available."""
    app = FastAPI()
    app.include_router(predict_router)

    # Initialize app state with model-derived dimension
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.expected_feature_dimension = 5  # Different from default

    client = TestClient(app)

    # Request with 4 features should fail now (expecting 5)
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "Expected 5, got 4" in data["detail"]

    # Request with 5 features should succeed
    payload = {
        "features": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 200
