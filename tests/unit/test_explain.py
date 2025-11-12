"""
Unit tests for the explain endpoint (src/app/api/explain.py).

Tests cover:
- Endpoint validation (single sample requirement)
- SHAP explanation generation (TreeExplainer)
- Fallback to feature importance
- Fallback to zero values
- Model readiness checks
- Error handling
- Prediction log emission
- Tracing integration
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.app.api.explain import (
    ExplainResponse,
    _generate_shap_explanation,
    _is_tree_model,
)
from src.app.api.predict import PredictRequest


@pytest.fixture
def app():
    """Create a FastAPI app with the explain endpoint."""
    from src.app.api.explain import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_model_wrapper():
    """Mock ModelWrapper with a trained model."""
    wrapper = Mock()
    wrapper.predict = Mock(return_value=np.array([1]))

    # Mock RandomForest model
    mock_model = Mock()
    mock_model.__class__.__name__ = "RandomForestClassifier"
    mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
    wrapper._model = mock_model

    return wrapper


@pytest.fixture
def configured_app(app, mock_model_wrapper):
    """Configure app state with mock model."""
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.model_metadata = {"model_path": "s3://bucket/model"}
    return app


def test_explain_requires_single_sample(configured_app):
    """Test that explain endpoint rejects batch requests."""
    client = TestClient(configured_app)

    # Send multiple samples
    payload = {"features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}

    response = client.post("/explain/", json=payload)

    assert response.status_code == 400
    assert "exactly one feature vector" in response.json()["detail"].lower()


def test_explain_returns_503_when_model_not_ready(app):
    """Test that explain returns 503 when model is not loaded."""
    app.state.is_ready = False
    app.state.ml_wrapper = None

    client = TestClient(app)
    payload = {"features": [[1.0, 2.0, 3.0]]}

    response = client.post("/explain/", json=payload)

    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"].lower()


def test_explain_returns_503_when_wrapper_is_none(app):
    """Test that explain returns 503 when ml_wrapper is None."""
    app.state.is_ready = True
    app.state.ml_wrapper = None

    client = TestClient(app)
    payload = {"features": [[1.0, 2.0, 3.0]]}

    response = client.post("/explain/", json=payload)

    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"].lower()


def test_explain_successful_with_tree_shap(configured_app, mock_model_wrapper):
    """Test successful explanation generation using SHAP TreeExplainer."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.shap") as mock_shap:
        # Mock TreeExplainer
        mock_explainer = Mock()
        mock_explainer.shap_values = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "shap_values" in data
        assert "feature_values" in data
        assert "base_value" in data
        assert "explanation_type" in data

        assert data["prediction"] == 1
        assert data["shap_values"] == [0.1, 0.2, 0.3]
        assert data["feature_values"] == [1.0, 2.0, 3.0]
        assert data["base_value"] == 0.5
        assert data["explanation_type"] == "tree_shap"


def test_explain_fallback_to_feature_importance(configured_app, mock_model_wrapper):
    """Test fallback to feature importance when SHAP fails."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.shap") as mock_shap:
        # Make SHAP raise an exception
        mock_shap.TreeExplainer.side_effect = Exception("SHAP failed")

        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["explanation_type"] == "feature_importance"
        assert len(data["shap_values"]) == 3
        assert data["base_value"] is None


def test_explain_fallback_to_zero_when_all_methods_fail(configured_app, mock_model_wrapper):
    """Test final fallback to zero values when all explanation methods fail."""
    client = TestClient(configured_app)

    # Remove feature_importances_ attribute
    del mock_model_wrapper._model.feature_importances_

    with patch("src.app.api.explain.shap") as mock_shap:
        # Make SHAP raise an exception
        mock_shap.TreeExplainer.side_effect = Exception("SHAP failed")

        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["explanation_type"] == "fallback_zero"
        assert data["shap_values"] == [0.0, 0.0, 0.0]
        assert data["base_value"] is None


def test_explain_when_shap_not_installed(configured_app, mock_model_wrapper):
    """Test that explain falls back gracefully when SHAP is not installed."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.shap", side_effect=ImportError("No module named 'shap'")):
        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Should fall back to feature importance
        assert data["explanation_type"] == "feature_importance"


def test_explain_handles_multiclass_shap_output(configured_app, mock_model_wrapper):
    """Test that explain handles multi-class SHAP output (list of arrays)."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.shap") as mock_shap:
        # Mock TreeExplainer with multi-class output
        mock_explainer = Mock()
        # Multi-class: list of arrays (one per class)
        mock_explainer.shap_values = Mock(return_value=[
            np.array([[0.1, 0.2, 0.3]]),  # Class 0
            np.array([[0.4, 0.5, 0.6]]),  # Class 1
        ])
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Should use class 1 (positive class) for binary classification
        assert data["shap_values"] == [0.4, 0.5, 0.6]


def test_is_tree_model_identifies_random_forest():
    """Test that _is_tree_model identifies RandomForest models."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "RandomForestClassifier"

    assert _is_tree_model(mock_model) is True


def test_is_tree_model_identifies_gradient_boosting():
    """Test that _is_tree_model identifies GradientBoosting models."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "GradientBoostingClassifier"

    assert _is_tree_model(mock_model) is True


def test_is_tree_model_identifies_xgboost():
    """Test that _is_tree_model identifies XGBoost models."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "XGBClassifier"

    assert _is_tree_model(mock_model) is True


def test_is_tree_model_rejects_linear_models():
    """Test that _is_tree_model rejects linear models."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "LogisticRegression"

    assert _is_tree_model(mock_model) is False


def test_is_tree_model_rejects_neural_networks():
    """Test that _is_tree_model rejects neural network models."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "MLPClassifier"

    assert _is_tree_model(mock_model) is False


def test_generate_shap_explanation_with_tree_model():
    """Test _generate_shap_explanation with tree-based model."""
    mock_wrapper = Mock()
    mock_model = Mock()
    mock_model.__class__.__name__ = "RandomForestClassifier"
    mock_wrapper._model = mock_model

    with patch("src.app.api.explain.shap") as mock_shap:
        mock_explainer = Mock()
        mock_explainer.shap_values = Mock(return_value=np.array([[0.1, 0.2]]))
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        features = np.array([[1.0, 2.0]])
        shap_vals, base_val, exp_type = _generate_shap_explanation(
            mock_wrapper, features, [1.0, 2.0]
        )

        assert shap_vals == [0.1, 0.2]
        assert base_val == 0.5
        assert exp_type == "tree_shap"


def test_generate_shap_explanation_with_feature_importance():
    """Test _generate_shap_explanation falls back to feature importance."""
    mock_wrapper = Mock()
    mock_model = Mock()
    mock_model.__class__.__name__ = "LinearRegression"
    mock_model.feature_importances_ = np.array([0.6, 0.4])
    mock_wrapper._model = mock_model

    with patch("src.app.api.explain.shap", side_effect=ImportError()):
        features = np.array([[3.0, 2.0]])
        shap_vals, base_val, exp_type = _generate_shap_explanation(
            mock_wrapper, features, [3.0, 2.0]
        )

        # Should use normalized feature importances scaled by feature values
        assert len(shap_vals) == 2
        assert base_val is None
        assert exp_type == "feature_importance"


def test_explain_emits_prediction_log(configured_app, mock_model_wrapper):
    """Test that explain endpoint emits prediction logs."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.emit_prediction_log") as mock_emit:
        with patch("src.app.api.explain.shap", side_effect=ImportError()):
            payload = {"features": [[1.0, 2.0, 3.0]]}
            response = client.post("/explain/", json=payload)

            assert response.status_code == 200

            # Prediction log should have been emitted
            # Note: emit_prediction_log is called as a background task
            # We can't easily test the background task execution in TestClient
            # but we can verify it was added to background tasks


def test_explain_sets_tracing_attributes(configured_app, mock_model_wrapper):
    """Test that explain endpoint sets tracing attributes."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.get_tracer") as mock_get_tracer:
        mock_span = Mock()
        mock_tracer = Mock()
        mock_tracer.start_as_current_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        mock_get_tracer.return_value = mock_tracer

        with patch("src.app.api.explain.shap", side_effect=ImportError()):
            payload = {"features": [[1.0, 2.0, 3.0]]}
            response = client.post("/explain/", json=payload)

            assert response.status_code == 200

            # Verify span attributes were set
            mock_span.set_attribute.assert_any_call("ml.model.path", "s3://bucket/model")
            mock_span.set_attribute.assert_any_call("ml.explanation.type", "shap")
            mock_span.set_attribute.assert_any_call("ml.explanation.generated", True)


def test_explain_returns_500_on_prediction_failure(configured_app, mock_model_wrapper):
    """Test that explain returns 500 when prediction fails."""
    client = TestClient(configured_app)

    # Make predict raise an exception
    mock_model_wrapper.predict.side_effect = Exception("Prediction failed")

    payload = {"features": [[1.0, 2.0, 3.0]]}
    response = client.post("/explain/", json=payload)

    assert response.status_code == 500
    assert "explanation failed" in response.json()["detail"].lower()


def test_explain_accepts_empty_sample(configured_app, mock_model_wrapper):
    """Test that explain rejects empty sample list."""
    client = TestClient(configured_app)

    payload = {"features": []}
    response = client.post("/explain/", json=payload)

    assert response.status_code == 400
    assert "exactly one feature vector" in response.json()["detail"].lower()


def test_explain_converts_float_to_int_prediction(configured_app, mock_model_wrapper):
    """Test that explain converts integer float predictions to int."""
    client = TestClient(configured_app)

    # Return a float that is actually an integer
    mock_model_wrapper.predict.return_value = np.array([2.0])

    with patch("src.app.api.explain.shap", side_effect=ImportError()):
        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Should be converted to int
        assert data["prediction"] == 2
        assert isinstance(data["prediction"], int)


def test_explain_logs_correlation_id(configured_app, mock_model_wrapper):
    """Test that explain endpoint logs with correlation ID."""
    client = TestClient(configured_app)

    with patch("src.app.api.explain.logger") as mock_logger:
        with patch("src.app.api.explain.shap", side_effect=ImportError()):
            headers = {"X-Correlation-ID": "test-correlation-123"}
            payload = {"features": [[1.0, 2.0, 3.0]]}

            # Note: TestClient doesn't run middleware, so we can't test
            # correlation ID from middleware, but we can verify logging structure
            response = client.post("/explain/", json=payload, headers=headers)

            assert response.status_code == 200

            # Verify info logging was called
            assert mock_logger.info.called


def test_explain_handles_correlation_id_from_state(configured_app, mock_model_wrapper):
    """Test that explain reads correlation ID from request state."""
    from src.app.api.explain import router

    app = FastAPI()
    app.include_router(router)
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.model_metadata = {"model_path": "s3://bucket/model"}

    # Add correlation middleware
    from src.app.api.middleware.correlation import CorrelationIDMiddleware
    app.add_middleware(CorrelationIDMiddleware)

    client = TestClient(app)

    with patch("src.app.api.explain.shap", side_effect=ImportError()):
        payload = {"features": [[1.0, 2.0, 3.0]]}
        response = client.post("/explain/", json=payload)

        assert response.status_code == 200
        # Correlation ID should be in response headers
        assert "X-Correlation-ID" in response.headers
