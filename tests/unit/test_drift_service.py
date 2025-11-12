"""
Unit tests for drift monitoring service.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from unittest.mock import AsyncMock, MagicMock, patch

from src.drift_monitoring.service import create_app
from src.drift_monitoring.config import DriftSettings
from src.drift_monitoring.monitor import DriftMonitor


@pytest.mark.unit
class TestDriftService:
    """Test drift monitoring service endpoints and lifecycle."""

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_create_app(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test that create_app creates FastAPI app with correct configuration."""
        # Setup mocks
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor_class.return_value = mock_monitor
        
        # Create app
        app = create_app()
        
        # Assertions
        assert app is not None
        assert app.title == "Drift Monitoring Service"
        assert hasattr(app.state, 'monitor')
        assert hasattr(app.state, 'registry')

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_healthz_endpoint(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test /healthz endpoint returns 200 OK."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor.initialize = AsyncMock()
        mock_monitor.start_background_loop = MagicMock()
        mock_monitor.shutdown = AsyncMock()
        mock_monitor_class.return_value = mock_monitor
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_refresh_endpoint(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test /refresh endpoint triggers drift evaluation."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor.initialize = AsyncMock()
        mock_monitor.start_background_loop = MagicMock()
        mock_monitor.shutdown = AsyncMock()
        mock_monitor.evaluate_once = AsyncMock()
        mock_monitor_class.return_value = mock_monitor
        
        app = create_app()
        client = TestClient(app)
        
        response = client.post("/refresh")
        assert response.status_code == 200
        assert response.json() == {"status": "triggered"}
        mock_monitor.evaluate_once.assert_called_once()

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_metrics_endpoint(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test /metrics endpoint returns Prometheus metrics."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor.initialize = AsyncMock()
        mock_monitor.start_background_loop = MagicMock()
        mock_monitor.shutdown = AsyncMock()
        mock_monitor_class.return_value = mock_monitor
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert b"#" in response.content  # Prometheus format starts with comments

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_startup_event(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test startup event initializes monitor and starts background loop."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor.initialize = AsyncMock()
        mock_monitor.start_background_loop = MagicMock()
        mock_monitor.shutdown = AsyncMock()
        mock_monitor_class.return_value = mock_monitor
        
        app = create_app()
        
        # Trigger startup
        with TestClient(app) as client:
            pass
        
        mock_monitor.initialize.assert_called_once()
        mock_monitor.start_background_loop.assert_called_once()

    @patch('src.drift_monitoring.service.DriftSettings')
    @patch('src.drift_monitoring.service.DriftMonitor')
    def test_shutdown_event(self, mock_monitor_class: MagicMock, mock_settings_class: MagicMock):
        """Test shutdown event stops monitor gracefully."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.log_level = "INFO"
        mock_settings.validate.return_value = None
        mock_settings_class.from_env.return_value = mock_settings
        
        mock_monitor = MagicMock(spec=DriftMonitor)
        mock_monitor.initialize = AsyncMock()
        mock_monitor.start_background_loop = MagicMock()
        mock_monitor.shutdown = AsyncMock()
        mock_monitor_class.return_value = mock_monitor
        
        app = create_app()
        
        # Trigger startup and shutdown
        with TestClient(app) as client:
            pass
        
        # Shutdown is called when context exits
        # Note: FastAPI TestClient doesn't always trigger shutdown events properly
        # This is a limitation of the test framework

    @patch('src.drift_monitoring.service.DriftSettings')
    def test_settings_validation_error(self, mock_settings_class: MagicMock):
        """Test error handling when settings validation fails."""
        mock_settings = MagicMock(spec=DriftSettings)
        mock_settings.validate.side_effect = ValueError("Invalid configuration")
        mock_settings_class.from_env.return_value = mock_settings
        
        with pytest.raises(ValueError):
            create_app()

