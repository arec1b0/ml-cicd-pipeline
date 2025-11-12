"""
Unit tests for explain endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api.explain import router as explain_router
from src.models.infer import ModelWrapper


@pytest.mark.unit
class TestExplainEndpoint:
    """Test explain endpoint functionality."""

    @pytest.fixture
    def mock_model_wrapper(self) -> MagicMock:
        """Create mock model wrapper."""
        wrapper = MagicMock(spec=ModelWrapper)
        wrapper.predict.return_value = [0]
        wrapper._model = MagicMock()
        wrapper._model.feature_importances_ = np.array([0.25, 0.25, 0.25, 0.25])
        return wrapper

    @pytest.fixture
    def test_app(self, mock_model_wrapper: MagicMock) -> FastAPI:
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(explain_router)
        
        app.state.is_ready = True
        app.state.ml_wrapper = mock_model_wrapper
        app.state.model_metadata = {"model_path": "test://model"}
        
        return app

    def test_explain_with_valid_single_sample(self, test_app: FastAPI, mock_model_wrapper: MagicMock):
        """Test /explain/ endpoint with valid single sample."""
        client = TestClient(test_app)
        
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        
        with patch('src.app.api.explain._generate_shap_explanation') as mock_shap:
            mock_shap.return_value = ([0.1, 0.2, 0.3, 0.4], 0.5, "tree_shap")
            
            response = client.post("/explain/", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "shap_values" in data
            assert "feature_values" in data
            assert "explanation_type" in data
            assert len(data["shap_values"]) == 4

    def test_explain_rejects_multiple_samples(self, test_app: FastAPI):
        """Test /explain/ endpoint rejects multiple samples."""
        client = TestClient(test_app)
        
        payload = {
            "features": [
                [5.1, 3.5, 1.4, 0.2],
                [6.7, 3.0, 5.2, 2.3],
            ]
        }
        
        response = client.post("/explain/", json=payload)
        
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()

    def test_explain_returns_503_when_model_not_loaded(self, mock_model_wrapper: MagicMock):
        """Test /explain/ endpoint returns 503 when model not loaded."""
        app = FastAPI()
        app.include_router(explain_router)
        
        app.state.is_ready = False
        app.state.ml_wrapper = None
        
        client = TestClient(app)
        
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        
        response = client.post("/explain/", json=payload)
        
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()

    def test_generate_shap_explanation_uses_tree_explainer(self, mock_model_wrapper: MagicMock):
        """Test _generate_shap_explanation uses TreeExplainer for tree models."""
        from src.app.api.explain import _generate_shap_explanation, _is_tree_model
        
        # Mock tree model
        mock_model_wrapper._model.__class__.__name__ = "RandomForestClassifier"
        
        with patch('shap.TreeExplainer') as mock_explainer_class:
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
            mock_explainer.expected_value = 0.5
            mock_explainer_class.return_value = mock_explainer
            
            features = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
            original_features = [5.1, 3.5, 1.4, 0.2]
            
            shap_values, base_value, explanation_type = _generate_shap_explanation(
                mock_model_wrapper, features, original_features
            )
            
            assert explanation_type == "tree_shap"
            assert base_value == 0.5
            assert len(shap_values) == 4

    def test_generate_shap_explanation_falls_back_to_feature_importance(self, mock_model_wrapper: MagicMock):
        """Test _generate_shap_explanation falls back to feature importance."""
        from src.app.api.explain import _generate_shap_explanation
        
        # Mock non-tree model
        mock_model_wrapper._model.__class__.__name__ = "SVC"
        
        features = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        original_features = [5.1, 3.5, 1.4, 0.2]
        
        with patch('shap.TreeExplainer', side_effect=ImportError("SHAP not available")):
            shap_values, base_value, explanation_type = _generate_shap_explanation(
                mock_model_wrapper, features, original_features
            )
            
            assert explanation_type == "feature_importance"
            assert base_value is None
            assert len(shap_values) == 4

    def test_generate_shap_explanation_returns_zeros_when_all_fail(self, mock_model_wrapper: MagicMock):
        """Test _generate_shap_explanation returns zeros when all methods fail."""
        from src.app.api.explain import _generate_shap_explanation
        
        # Mock model without feature_importances_
        mock_model_wrapper._model.__class__.__name__ = "SVC"
        del mock_model_wrapper._model.feature_importances_
        
        features = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        original_features = [5.1, 3.5, 1.4, 0.2]
        
        with patch('shap.TreeExplainer', side_effect=ImportError("SHAP not available")):
            shap_values, base_value, explanation_type = _generate_shap_explanation(
                mock_model_wrapper, features, original_features
            )
            
            assert explanation_type == "fallback_zero"
            assert base_value is None
            assert all(v == 0.0 for v in shap_values)

    def test_is_tree_model_identifies_tree_models(self):
        """Test _is_tree_model correctly identifies tree-based models."""
        from src.app.api.explain import _is_tree_model
        
        tree_models = [
            "RandomForestClassifier",
            "XGBClassifier",
            "LGBMClassifier",
            "DecisionTreeClassifier",
        ]
        
        for model_name in tree_models:
            mock_model = MagicMock()
            mock_model.__class__.__name__ = model_name
            assert _is_tree_model(mock_model) is True
        
        non_tree_models = [
            "SVC",
            "LogisticRegression",
            "KNeighborsClassifier",
        ]
        
        for model_name in non_tree_models:
            mock_model = MagicMock()
            mock_model.__class__.__name__ = model_name
            assert _is_tree_model(mock_model) is False

    def test_correlation_id_propagation(self, test_app: FastAPI, mock_model_wrapper: MagicMock):
        """Test correlation ID propagation."""
        client = TestClient(test_app)
        
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        
        with patch('src.app.api.explain._generate_shap_explanation') as mock_shap:
            mock_shap.return_value = ([0.1, 0.2, 0.3, 0.4], 0.5, "tree_shap")
            
            response = client.post(
                "/explain/",
                json=payload,
                headers={"X-Correlation-ID": "test-correlation-id"}
            )
            
            assert response.status_code == 200
            # Correlation ID should be in response headers
            assert "X-Correlation-ID" in response.headers

