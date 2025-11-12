"""
Unit tests for the drift monitoring service (src/drift_monitoring/service.py).

Tests cover:
- App creation and configuration
- Startup/shutdown lifecycle
- Health endpoint
- Refresh endpoint
- Metrics endpoint
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry


@pytest.fixture
def mock_settings():
    """Mock DriftSettings object with valid configuration."""
    with patch("src.drift_monitoring.service.DriftSettings") as mock_settings_class:
        settings = Mock()
        settings.reference_dataset_uri = "s3://bucket/train.csv"
        settings.current_dataset_uri = "s3://bucket/current.csv"
        settings.loki_base_url = None
        settings.loki_query = None
        settings.loki_range_minutes = 60
        settings.evaluation_interval_seconds = 300
        settings.min_rows = 50
        settings.max_rows = 1000
        settings.log_level = "INFO"
        settings.validate = Mock()

        mock_settings_class.from_env.return_value = settings
        yield settings


@pytest.fixture
def mock_monitor():
    """Mock DriftMonitor object."""
    with patch("src.drift_monitoring.service.DriftMonitor") as mock_monitor_class:
        monitor = Mock()
        monitor.initialize = AsyncMock()
        monitor.start_background_loop = Mock()
        monitor.shutdown = AsyncMock()
        monitor.evaluate_once = AsyncMock()

        mock_monitor_class.return_value = monitor
        yield monitor


def test_create_app_initializes_settings(mock_settings, mock_monitor):
    """Test that create_app calls DriftSettings.from_env and validate."""
    from src.drift_monitoring.service import create_app

    app = create_app()

    # Verify settings were loaded and validated
    mock_settings.validate.assert_called_once()

    # Verify app was configured
    assert app.title == "Drift Monitoring Service"
    assert app.version == "0.1.0"


def test_create_app_creates_monitor(mock_settings, mock_monitor):
    """Test that create_app creates a DriftMonitor and stores it in app state."""
    from src.drift_monitoring.service import create_app

    app = create_app()

    # Verify monitor was created and stored
    assert hasattr(app.state, "monitor")
    assert app.state.monitor is mock_monitor

    # Verify registry was created and stored
    assert hasattr(app.state, "registry")
    assert isinstance(app.state.registry, CollectorRegistry)


def test_healthz_endpoint_returns_ok(mock_settings, mock_monitor):
    """Test that /healthz endpoint returns status ok."""
    from src.drift_monitoring.service import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_refresh_endpoint_triggers_evaluation(mock_settings, mock_monitor):
    """Test that /refresh endpoint calls monitor.evaluate_once."""
    from src.drift_monitoring.service import create_app

    app = create_app()
    client = TestClient(app)

    response = client.post("/refresh")

    assert response.status_code == 200
    assert response.json() == {"status": "triggered"}

    # Verify evaluate_once was called
    mock_monitor.evaluate_once.assert_called_once()


def test_metrics_endpoint_returns_prometheus_format(mock_settings, mock_monitor):
    """Test that /metrics endpoint returns Prometheus metrics."""
    from src.drift_monitoring.service import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Prometheus metrics format should be plain text
    assert isinstance(response.content, bytes)


def test_startup_event_initializes_monitor(mock_settings, mock_monitor):
    """Test that startup event calls monitor.initialize and starts background loop."""
    from src.drift_monitoring.service import create_app

    app = create_app()

    # Use TestClient which triggers startup/shutdown events
    with TestClient(app):
        # Verify initialization was called
        mock_monitor.initialize.assert_called_once()
        mock_monitor.start_background_loop.assert_called_once()


def test_shutdown_event_stops_monitor(mock_settings, mock_monitor):
    """Test that shutdown event calls monitor.shutdown."""
    from src.drift_monitoring.service import create_app

    app = create_app()

    # Use TestClient context manager which triggers shutdown on exit
    with TestClient(app):
        pass

    # Verify shutdown was called
    mock_monitor.shutdown.assert_called_once()


def test_logging_level_configured_from_settings(mock_settings, mock_monitor):
    """Test that logging level is configured from settings."""
    from src.drift_monitoring.service import create_app

    mock_settings.log_level = "DEBUG"

    with patch("src.drift_monitoring.service.logging.basicConfig") as mock_basic_config:
        app = create_app()

        # Verify logging was configured with DEBUG level
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == 10  # logging.DEBUG = 10


def test_create_app_handles_invalid_log_level(mock_settings, mock_monitor):
    """Test that invalid log level falls back to INFO."""
    from src.drift_monitoring.service import create_app

    mock_settings.log_level = "INVALID"

    with patch("src.drift_monitoring.service.logging.basicConfig") as mock_basic_config:
        app = create_app()

        # Verify logging was configured with INFO level (default)
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        # getattr with fallback should return logging.INFO = 20
        assert call_args[1]["level"] == 20


def test_app_state_contains_registry(mock_settings, mock_monitor):
    """Test that app state contains the Prometheus registry."""
    from src.drift_monitoring.service import create_app

    app = create_app()

    assert hasattr(app.state, "registry")
    assert isinstance(app.state.registry, CollectorRegistry)


def test_refresh_endpoint_handles_evaluation_error(mock_settings, mock_monitor):
    """Test that /refresh endpoint handles errors during evaluation."""
    from src.drift_monitoring.service import create_app

    # Make evaluate_once raise an exception
    mock_monitor.evaluate_once.side_effect = Exception("Evaluation failed")

    app = create_app()
    client = TestClient(app)

    # The endpoint should propagate the exception (FastAPI will convert to 500)
    with pytest.raises(Exception, match="Evaluation failed"):
        response = client.post("/refresh")


def test_monitor_receives_correct_settings(mock_settings, mock_monitor):
    """Test that DriftMonitor is initialized with correct settings and registry."""
    from src.drift_monitoring.service import create_app, DriftMonitor

    app = create_app()

    # Verify DriftMonitor was called with settings and registry
    DriftMonitor.assert_called_once()
    call_args = DriftMonitor.call_args
    assert call_args[0][0] is mock_settings
    assert isinstance(call_args[0][1], CollectorRegistry)
