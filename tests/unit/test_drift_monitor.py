"""
Unit tests for drift monitor core logic.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from src.drift_monitoring.monitor import DriftMonitor, DriftMetrics
from src.drift_monitoring.config import DriftSettings


@pytest.mark.unit
class TestDriftMonitor:
    """Test drift monitor initialization and evaluation."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock drift settings."""
        settings = MagicMock(spec=DriftSettings)
        settings.reference_dataset_uri = "file:///test/reference.csv"
        settings.current_dataset_uri = "file:///test/current.csv"
        settings.loki_base_url = None
        settings.loki_query = None
        settings.loki_range_minutes = 60
        settings.evaluation_interval_seconds = 300
        settings.min_rows = 50
        settings.max_rows = None
        settings.log_level = "INFO"
        return settings

    @pytest.fixture
    def registry(self) -> CollectorRegistry:
        """Create Prometheus registry."""
        return CollectorRegistry()

    @pytest.fixture
    def reference_df(self) -> pd.DataFrame:
        """Create reference dataframe."""
        return pd.DataFrame({
            'feature_0': [5.1, 6.7, 5.0, 6.3, 5.5],
            'feature_1': [3.5, 3.0, 3.3, 3.3, 3.2],
            'feature_2': [1.4, 5.2, 1.4, 6.0, 1.5],
            'feature_3': [0.2, 2.3, 0.2, 2.5, 0.3],
            'target': [0, 1, 0, 1, 0],
            'prediction': [0, 1, 0, 1, 0],
        })

    def test_init_creates_metrics(self, mock_settings: MagicMock, registry: CollectorRegistry):
        """Test that __init__ creates Prometheus metrics."""
        monitor = DriftMonitor(mock_settings, registry)
        
        assert monitor.settings == mock_settings
        assert monitor.registry == registry
        assert monitor.data_drift_status is not None
        assert monitor.data_drift_share is not None
        assert monitor.prediction_drift_status is not None

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_initialize_loads_reference_dataset(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry,
        reference_df: pd.DataFrame
    ):
        """Test that initialize loads reference dataset and sets up column mapping."""
        mock_load.return_value = reference_df
        
        monitor = DriftMonitor(mock_settings, registry)
        
        async def run_init():
            await monitor.initialize()
        
        asyncio.run(run_init())
        
        assert monitor.reference_df is not None
        assert len(monitor.feature_columns) == 4
        assert monitor.prediction_column == "prediction"
        assert monitor.target_column == "target"
        assert monitor.column_mapping is not None

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_initialize_raises_value_error_empty_dataset(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry
    ):
        """Test that initialize raises ValueError for empty dataset."""
        mock_load.return_value = pd.DataFrame()
        
        monitor = DriftMonitor(mock_settings, registry)
        
        async def run_init():
            with pytest.raises(ValueError, match="empty"):
                await monitor.initialize()
        
        asyncio.run(run_init())

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_initialize_raises_value_error_no_features(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry
    ):
        """Test that initialize raises ValueError for missing feature columns."""
        mock_load.return_value = pd.DataFrame({'target': [0, 1], 'prediction': [0, 1]})
        
        monitor = DriftMonitor(mock_settings, registry)
        
        async def run_init():
            with pytest.raises(ValueError, match="feature columns"):
                await monitor.initialize()
        
        asyncio.run(run_init())

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_evaluate_once_raises_runtime_error_not_initialized(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry
    ):
        """Test that evaluate_once raises RuntimeError when not initialized."""
        monitor = DriftMonitor(mock_settings, registry)
        
        async def run_eval():
            with pytest.raises(RuntimeError, match="initialized"):
                await monitor.evaluate_once()
        
        asyncio.run(run_eval())

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_evaluate_once_skips_insufficient_data(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry,
        reference_df: pd.DataFrame
    ):
        """Test that evaluate_once skips evaluation when insufficient data."""
        mock_load.return_value = reference_df
        
        monitor = DriftMonitor(mock_settings, registry)
        monitor.settings.min_rows = 100  # Require more rows than available
        
        async def run_test():
            await monitor.initialize()
            # Mock _load_current_dataframe to return small dataset
            monitor._load_current_dataframe = lambda: pd.DataFrame({
                'feature_0': [5.1],
                'feature_1': [3.5],
                'feature_2': [1.4],
                'feature_3': [0.2],
            })
            
            await monitor.evaluate_once()
        
        asyncio.run(run_test())
        
        # Should not raise, just skip

    def test_extract_metrics_parses_report(self, mock_settings: MagicMock, registry: CollectorRegistry):
        """Test that _extract_metrics parses Evidently report correctly."""
        monitor = DriftMonitor(mock_settings, registry)
        
        report_dict = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "dataset_drift": True,
                        "share_of_drifted_columns": 0.5,
                    },
                },
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "drift_by_columns": {
                            "feature_0": {"drift_detected": True, "drift_score": 0.8},
                            "feature_1": {"drift_detected": False, "drift_score": 0.2},
                        },
                    },
                },
                {
                    "metric": "ColumnDriftMetric",
                    "result": {
                        "column_name": "prediction",
                        "drift_detected": True,
                        "drift_score": 0.7,
                    },
                },
            ],
        }
        
        metrics = monitor._extract_metrics(report_dict, 100, 0.5)
        
        assert metrics.data_drift_detected is True
        assert metrics.data_drift_share == 0.5
        assert metrics.prediction_drift_detected is True
        assert metrics.prediction_drift_score == 0.7
        assert metrics.prediction_psi == 0.5
        assert len(metrics.feature_metrics) == 2

    def test_update_prometheus_updates_metrics(
        self,
        mock_settings: MagicMock,
        registry: CollectorRegistry,
        reference_df: pd.DataFrame
    ):
        """Test that _update_prometheus updates all metrics."""
        monitor = DriftMonitor(mock_settings, registry)
        monitor.reference_df = reference_df
        
        metrics = DriftMetrics(
            data_drift_detected=True,
            data_drift_share=0.5,
            prediction_drift_detected=True,
            prediction_drift_score=0.7,
            prediction_psi=0.5,
            feature_metrics=[("feature_0", True, 0.8)],
            current_rows=100,
            evaluated_at=datetime.now(timezone.utc),
        )
        
        monitor._update_prometheus(metrics)
        
        assert monitor.data_drift_status._value.get() == 1
        assert monitor.data_drift_share._value.get() == 0.5
        assert monitor.prediction_drift_status._value.get() == 1

    def test_compute_prediction_psi(self, mock_settings: MagicMock, registry: CollectorRegistry):
        """Test that _compute_prediction_psi calculates PSI correctly."""
        monitor = DriftMonitor(mock_settings, registry)
        monitor.reference_df = pd.DataFrame({
            'prediction': [0, 0, 1, 1, 0],
        })
        monitor.prediction_column = "prediction"
        
        current_df = pd.DataFrame({
            'prediction': [0, 1, 1, 1, 0],
        })
        
        psi = monitor._compute_prediction_psi(current_df)
        
        assert psi is not None
        assert psi >= 0

    def test_sanitize_loki_query_prevents_injection(self, mock_settings: MagicMock, registry: CollectorRegistry):
        """Test that _sanitize_loki_query prevents injection attacks."""
        monitor = DriftMonitor(mock_settings, registry)
        
        # Test dangerous characters
        dangerous_queries = [
            "query; DROP TABLE",
            "query & malicious",
            "query | command",
            "query `backtick`",
            "query $(command)",
        ]
        
        for query in dangerous_queries:
            with pytest.raises(ValueError, match="dangerous"):
                monitor._sanitize_loki_query(query)
        
        # Test long query
        long_query = "a" * 1001
        with pytest.raises(ValueError, match="maximum length"):
            monitor._sanitize_loki_query(long_query)
        
        # Test valid query
        valid_query = "query{job=\"ml-predictions\"}"
        result = monitor._sanitize_loki_query(valid_query)
        assert result == valid_query

    def test_events_to_dataframe(self, mock_settings: MagicMock, registry: CollectorRegistry):
        """Test that _events_to_dataframe converts events correctly."""
        monitor = DriftMonitor(mock_settings, registry)
        monitor.feature_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        monitor.prediction_column = "prediction"
        monitor.target_column = "target"
        
        events = [
            {
                "features": [[5.1, 3.5, 1.4, 0.2]],
                "predictions": [0],
                "targets": [0],
            },
            {
                "features": [[6.7, 3.0, 5.2, 2.3]],
                "predictions": [1],
                "targets": [1],
            },
        ]
        
        df = monitor._events_to_dataframe(events)
        
        assert len(df) == 2
        assert 'feature_0' in df.columns
        assert 'prediction' in df.columns
        assert 'target' in df.columns

    @patch('src.drift_monitoring.monitor.load_dataset_from_uri')
    def test_shutdown_cancels_task(
        self,
        mock_load: MagicMock,
        mock_settings: MagicMock,
        registry: CollectorRegistry,
        reference_df: pd.DataFrame
    ):
        """Test that shutdown cancels background task."""
        mock_load.return_value = reference_df
        
        monitor = DriftMonitor(mock_settings, registry)
        
        async def run_test():
            await monitor.initialize()
            monitor.start_background_loop()
            
            # Wait a bit then shutdown
            await asyncio.sleep(0.1)
            await monitor.shutdown()
        
        asyncio.run(run_test())
        
        assert monitor._shutdown_event.is_set()

