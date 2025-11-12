"""
Integration tests for Loki integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.drift_monitoring.monitor import DriftMonitor
from src.drift_monitoring.config import DriftSettings


@pytest.mark.integration
class TestLokiIntegration:
    """Test Loki integration functionality."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock drift settings with Loki config."""
        settings = MagicMock(spec=DriftSettings)
        settings.reference_dataset_uri = "file:///test/reference.csv"
        settings.current_dataset_uri = None
        settings.loki_base_url = "http://localhost:3100"
        settings.loki_query = '{job="ml-predictions"}'
        settings.loki_range_minutes = 60
        settings.evaluation_interval_seconds = 300
        settings.min_rows = 50
        settings.max_rows = None
        settings.log_level = "INFO"
        return settings

    @patch('src.drift_monitoring.monitor.requests.get')
    def test_drift_monitor_queries_loki_successfully(
        self,
        mock_get: MagicMock,
        mock_settings: MagicMock,
    ):
        """Test drift monitor queries Loki successfully."""
        from prometheus_client import CollectorRegistry
        
        # Mock Loki response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {"job": "ml-predictions"},
                        "values": [
                            [
                                "1234567890000000000",
                                '{"features": [[5.1, 3.5, 1.4, 0.2]], "predictions": [0]}',
                            ],
                        ],
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        registry = CollectorRegistry()
        monitor = DriftMonitor(mock_settings, registry)
        monitor.feature_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        monitor.prediction_column = "prediction"
        
        events = monitor._collect_from_loki()
        
        assert len(events) > 0
        mock_get.assert_called_once()

    def test_loki_query_sanitization_prevents_injection(self, mock_settings: MagicMock):
        """Test Loki query sanitization prevents injection."""
        from prometheus_client import CollectorRegistry
        
        registry = CollectorRegistry()
        monitor = DriftMonitor(mock_settings, registry)
        
        # Test dangerous queries
        dangerous_queries = [
            "query; DROP TABLE",
            "query & malicious",
            "query | command",
        ]
        
        for query in dangerous_queries:
            with pytest.raises(ValueError, match="dangerous"):
                monitor._sanitize_loki_query(query)

    @patch('src.drift_monitoring.monitor.requests.get')
    def test_event_parsing_from_loki_logs(
        self,
        mock_get: MagicMock,
        mock_settings: MagicMock,
    ):
        """Test event parsing from Loki logs."""
        from prometheus_client import CollectorRegistry
        
        # Mock Loki response with multiple events
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {"job": "ml-predictions"},
                        "values": [
                            [
                                "1234567890000000000",
                                '{"features": [[5.1, 3.5, 1.4, 0.2]], "predictions": [0]}',
                            ],
                            [
                                "1234567891000000000",
                                '{"features": [[6.7, 3.0, 5.2, 2.3]], "predictions": [1]}',
                            ],
                        ],
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        registry = CollectorRegistry()
        monitor = DriftMonitor(mock_settings, registry)
        monitor.feature_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        monitor.prediction_column = "prediction"
        
        events = monitor._collect_from_loki()
        
        assert len(events) == 2
        assert "features" in events[0]
        assert "predictions" in events[0]

    @patch('src.drift_monitoring.monitor.requests.get')
    def test_error_handling_when_loki_unavailable(
        self,
        mock_get: MagicMock,
        mock_settings: MagicMock,
    ):
        """Test error handling when Loki unavailable."""
        from prometheus_client import CollectorRegistry
        
        # Mock connection error
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        registry = CollectorRegistry()
        monitor = DriftMonitor(mock_settings, registry)
        monitor.feature_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        
        # Should handle error gracefully
        with pytest.raises(requests.ConnectionError):
            events = monitor._collect_from_loki()

