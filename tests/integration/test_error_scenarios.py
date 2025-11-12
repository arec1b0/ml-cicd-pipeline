"""
Integration tests for error scenarios.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from src.app.main import create_app
from src.models.infer import ModelWrapper


@pytest.mark.integration
class TestErrorScenarios:
    """Test error handling scenarios."""

    @patch('src.app.main.ModelManager')
    def test_mlflow_unavailable_during_startup(self, mock_manager_class: MagicMock):
        """Test MLflow unavailable during startup."""
        mock_manager = MagicMock()
        mock_manager.supports_auto_refresh = False
        mock_manager.reload = AsyncMock(side_effect=MlflowException("Connection failed"))
        mock_manager_class.return_value = mock_manager
        
        app = create_app()
        
        # Should not raise exception, but model should not be ready
        with TestClient(app) as client:
            assert app.state.is_ready is False
            assert app.state.mlflow_connectivity["status"] == "error"

    @patch('src.app.main.ModelManager')
    def test_mlflow_unavailable_during_prediction(self, mock_manager_class: MagicMock):
        """Test MLflow unavailable during prediction."""
        mock_model_wrapper = MagicMock(spec=ModelWrapper)
        mock_model_wrapper.predict.return_value = [0, 1]
        mock_model_wrapper.get_input_dimension.return_value = 4
        
        mock_manager = MagicMock()
        mock_manager.supports_auto_refresh = False
        mock_manager.reload = AsyncMock(return_value=None)
        mock_manager_class.return_value = mock_manager
        
        app = create_app()
        
        # Manually set model for testing
        app.state.is_ready = True
        app.state.ml_wrapper = mock_model_wrapper
        app.state.expected_feature_dimension = 4
        
        client = TestClient(app)
        
        # Prediction should work even if MLflow was unavailable at startup
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        response = client.post("/predict/", json=payload)
        
        assert response.status_code == 200

    def test_malformed_json_input(self):
        """Test malformed JSON input."""
        app = create_app()
        client = TestClient(app)
        
        # Send invalid JSON
        response = client.post(
            "/predict/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_oversized_payloads(self):
        """Test oversized payloads."""
        app = create_app()
        client = TestClient(app)
        
        # Create very large payload
        large_features = [[1.0] * 4] * 10000
        
        payload = {"features": large_features}
        
        response = client.post("/predict/", json=payload)
        
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 400, 413]

    def test_invalid_feature_types(self):
        """Test invalid feature types."""
        app = create_app()
        client = TestClient(app)
        
        # Set model state
        mock_wrapper = MagicMock(spec=ModelWrapper)
        mock_wrapper.get_input_dimension.return_value = 4
        app.state.is_ready = True
        app.state.ml_wrapper = mock_wrapper
        app.state.expected_feature_dimension = 4
        
        # Send string instead of number
        payload = {"features": [["5.1", "3.5", "1.4", "0.2"]]}
        
        response = client.post("/predict/", json=payload)
        
        # Should return validation error
        assert response.status_code in [400, 422]

    def test_missing_required_headers(self):
        """Test missing required headers."""
        app = create_app()
        client = TestClient(app)
        
        # Admin endpoint requires header
        app.state.admin_api_token = "test-token"
        
        response = client.post("/admin/reload")
        
        # Should return 401 or 403
        assert response.status_code in [401, 403]

    @patch('src.app.main.ModelManager')
    def test_service_degradation_scenarios(self, mock_manager_class: MagicMock):
        """Test service degradation scenarios."""
        mock_manager = MagicMock()
        mock_manager.supports_auto_refresh = False
        mock_manager.reload = AsyncMock(side_effect=Exception("Unexpected error"))
        mock_manager_class.return_value = mock_manager
        
        app = create_app()
        
        # Should handle unexpected errors gracefully
        with TestClient(app) as client:
            # Service should still be available even if model loading fails
            response = client.get("/health/")
            assert response.status_code == 200

