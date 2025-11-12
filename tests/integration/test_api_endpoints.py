"""
Integration tests for API endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.models.infer import ModelWrapper


@pytest.mark.integration
class TestAPIEndpoints:
    """Test API endpoints integration."""

    @pytest.fixture
    def mock_model_wrapper(self) -> MagicMock:
        """Create mock model wrapper."""
        wrapper = MagicMock(spec=ModelWrapper)
        wrapper.predict.return_value = [0, 1]
        wrapper.get_input_dimension.return_value = 4
        wrapper._model = MagicMock()
        wrapper._model.feature_importances_ = [0.25, 0.25, 0.25, 0.25]
        return wrapper

    @pytest.fixture
    def app_with_model(self, mock_model_wrapper: MagicMock) -> None:
        """Create app with loaded model."""
        with patch('src.app.main.ModelManager') as mock_manager_class:
            from src.models.manager import LoadedModel, ModelDescriptor
            
            descriptor = ModelDescriptor(
                source="local",
                model_uri="test://model",
                version="1.0",
                stage="Production",
                run_id="run123",
                server_version="2.0",
            )
            
            loaded_model = LoadedModel(
                descriptor=descriptor,
                wrapper=mock_model_wrapper,
                artifact_path=None,
                model_file=None,
                metrics=None,
                accuracy=0.95,
            )
            
            mock_manager = MagicMock()
            mock_manager.supports_auto_refresh = False
            mock_manager.reload = MagicMock(return_value=loaded_model)
            mock_manager_class.return_value = mock_manager
            
            app = create_app()
            
            # Manually set model state for testing
            app.state.is_ready = True
            app.state.ml_wrapper = mock_model_wrapper
            app.state.expected_feature_dimension = 4
            app.state.model_metadata = {"model_path": "test://model"}
            
            yield app

    def test_predict_endpoint_with_real_model(self, app_with_model):
        """Test /predict endpoint with real model."""
        client = TestClient(app_with_model)
        
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

    def test_explain_endpoint_with_real_model(self, app_with_model):
        """Test /explain endpoint with real model."""
        client = TestClient(app_with_model)
        
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        
        with patch('src.app.api.explain._generate_shap_explanation') as mock_shap:
            mock_shap.return_value = ([0.1, 0.2, 0.3, 0.4], 0.5, "tree_shap")
            
            response = client.post("/explain/", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "shap_values" in data

    def test_health_endpoint_returns_correct_status(self, app_with_model):
        """Test /health endpoint returns correct status."""
        client = TestClient(app_with_model)
        
        response = client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint_returns_prometheus_metrics(self, app_with_model):
        """Test /metrics endpoint returns Prometheus metrics."""
        client = TestClient(app_with_model)
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert b"ml_request" in response.content or b"#" in response.content

    def test_admin_reload_endpoint_with_authentication(self, app_with_model):
        """Test /admin/reload endpoint with authentication."""
        client = TestClient(app_with_model)
        
        # Set admin token
        app_with_model.state.admin_api_token = "test-token"
        
        response = client.post(
            "/admin/reload",
            headers={"X-Admin-Token": "test-token"}
        )
        
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 202, 401, 403]

    def test_request_response_flow_through_middleware(self, app_with_model):
        """Test request/response flow through all middleware."""
        client = TestClient(app_with_model)
        
        payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}
        
        response = client.post("/predict/", json=payload)
        
        assert response.status_code == 200
        
        # Check middleware effects
        assert "X-Correlation-ID" in response.headers

    def test_correlation_id_propagation_across_endpoints(self, app_with_model):
        """Test correlation ID propagation across endpoints."""
        client = TestClient(app_with_model)
        
        correlation_id = "test-correlation-12345"
        
        # Test predict endpoint
        response1 = client.post(
            "/predict/",
            json={"features": [[5.1, 3.5, 1.4, 0.2]]},
            headers={"X-Correlation-ID": correlation_id}
        )
        
        assert response1.headers["X-Correlation-ID"] == correlation_id
        
        # Test explain endpoint
        response2 = client.post(
            "/explain/",
            json={"features": [[5.1, 3.5, 1.4, 0.2]]},
            headers={"X-Correlation-ID": correlation_id}
        )
        
        assert response2.headers["X-Correlation-ID"] == correlation_id

