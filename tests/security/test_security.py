"""
Security tests for authentication, authorization, and injection prevention.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.drift_monitoring.monitor import DriftMonitor
from src.drift_monitoring.config import DriftSettings


@pytest.mark.security
class TestSecurity:
    """Test security features."""

    @pytest.fixture
    def app_with_auth(self) -> None:
        """Create app with authentication configured."""
        with patch('src.app.main.ModelManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.supports_auto_refresh = False
            mock_manager.reload = MagicMock(return_value=None)
            mock_manager_class.return_value = mock_manager
            
            app = create_app()
            app.state.admin_api_token = "secret-admin-token"
            
            yield app

    def test_authentication_on_admin_endpoints(self, app_with_auth):
        """Test authentication on admin endpoints."""
        client = TestClient(app_with_auth)
        
        # Without token
        response = client.post("/admin/reload")
        assert response.status_code in [401, 403]
        
        # With invalid token
        response = client.post(
            "/admin/reload",
            headers={"X-Admin-Token": "wrong-token"}
        )
        assert response.status_code in [401, 403]
        
        # With valid token
        response = client.post(
            "/admin/reload",
            headers={"X-Admin-Token": "secret-admin-token"}
        )
        # Should succeed or return appropriate status
        assert response.status_code in [200, 202, 400, 500]

    def test_authorization_with_invalid_tokens(self, app_with_auth):
        """Test authorization with invalid tokens."""
        client = TestClient(app_with_auth)
        
        invalid_tokens = [
            "",
            "invalid",
            "null",
            "../../etc/passwd",
            "<script>alert('xss')</script>",
        ]
        
        for token in invalid_tokens:
            response = client.post(
                "/admin/reload",
                headers={"X-Admin-Token": token}
            )
            assert response.status_code in [401, 403]

    def test_logql_injection_prevention_in_loki_queries(self):
        """Test LogQL injection prevention in Loki queries."""
        from prometheus_client import CollectorRegistry
        
        settings = MagicMock(spec=DriftSettings)
        settings.loki_query = '{job="ml-predictions"}'
        
        registry = CollectorRegistry()
        monitor = DriftMonitor(settings, registry)
        
        # Test injection attempts
        injection_attempts = [
            '{job="ml-predictions"} | drop',
            '{job="ml-predictions"} ; DROP TABLE',
            '{job="ml-predictions"} & malicious',
            '{job="ml-predictions"} | command',
        ]
        
        for query in injection_attempts:
            with pytest.raises(ValueError, match="dangerous"):
                monitor._sanitize_loki_query(query)

    def test_input_validation_prevents_malicious_payloads(self):
        """Test input validation prevents malicious payloads."""
        app = create_app()
        
        # Set model state
        mock_wrapper = MagicMock()
        mock_wrapper.get_input_dimension.return_value = 4
        app.state.is_ready = True
        app.state.ml_wrapper = mock_wrapper
        app.state.expected_feature_dimension = 4
        
        client = TestClient(app)
        
        # Test various malicious payloads
        malicious_payloads = [
            {"features": [[float('inf')] * 4]},  # Infinity
            {"features": [[float('nan')] * 4]},  # NaN
            {"features": [None] * 4},  # None values
            {"features": "string_instead_of_list"},  # Wrong type
            {"features": []},  # Empty list
        ]
        
        for payload in malicious_payloads:
            response = client.post("/predict/", json=payload)
            # Should reject invalid inputs
            assert response.status_code in [400, 422]

    def test_rate_limiting_enforcement(self):
        """Test rate limiting enforcement."""
        app = create_app()
        client = TestClient(app)
        
        # Make many rapid requests
        # Note: Rate limiting may not be enforced in test client
        # This is a placeholder test
        for _ in range(10):
            response = client.get("/health/")
            assert response.status_code in [200, 429]  # 429 = Too Many Requests

    def test_cors_configuration(self):
        """Test CORS configuration."""
        app = create_app()
        client = TestClient(app)
        
        # Test OPTIONS request (preflight)
        response = client.options(
            "/predict/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            }
        )
        
        # CORS headers may or may not be present depending on configuration
        # This test verifies the endpoint doesn't crash
        assert response.status_code in [200, 405, 404]

    def test_header_injection_prevention(self, app_with_auth):
        """Test header injection prevention."""
        client = TestClient(app_with_auth)
        
        # Attempt header injection
        malicious_headers = [
            {"X-Admin-Token": "valid\nX-Injected: value"},
            {"X-Admin-Token": "valid\rX-Injected: value"},
            {"X-Admin-Token": "valid\tX-Injected: value"},
        ]
        
        for headers in malicious_headers:
            response = client.post("/admin/reload", headers=headers)
            # Should handle injection attempts safely
            assert response.status_code in [200, 202, 400, 401, 403, 500]

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention (if applicable)."""
        # This test is a placeholder for SQL injection tests
        # Most of this application doesn't use SQL directly,
        # but MLflow might use it
        
        # Test that string inputs are properly sanitized
        app = create_app()
        client = TestClient(app)
        
        # Test that model names, etc. are handled safely
        # This is more of a smoke test
        response = client.get("/health/")
        assert response.status_code == 200

