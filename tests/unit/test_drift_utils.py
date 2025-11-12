"""
Unit tests for drift utilities.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.drift import (
    load_dataset_from_uri,
    persist_reference_dataset,
    emit_prediction_log,
)


@pytest.mark.unit
class TestDriftUtils:
    """Test drift utility functions."""

    def test_load_dataset_from_uri_loads_local_csv(self, temp_csv_file: Path):
        """Test load_dataset_from_uri loads local CSV."""
        df = load_dataset_from_uri(str(temp_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'feature_0' in df.columns

    def test_load_dataset_from_uri_raises_file_not_found_error(self):
        """Test load_dataset_from_uri raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_dataset_from_uri("nonexistent_file.csv")

    @patch('pandas.read_csv')
    def test_load_dataset_from_uri_loads_remote_uri(self, mock_read_csv: pytest.Mock):
        """Test load_dataset_from_uri loads remote URI (mock fsspec)."""
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        df = load_dataset_from_uri("s3://bucket/data.csv")
        
        assert isinstance(df, pd.DataFrame)
        mock_read_csv.assert_called_once()

    def test_persist_reference_dataset_saves_to_file(self, temp_dir: Path):
        """Test persist_reference_dataset saves to file."""
        features = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
        targets = [0, 1]
        predictions = [0, 1]
        
        destination = str(temp_dir / "reference.csv")
        
        result = persist_reference_dataset(
            features=features,
            targets=targets,
            predictions=predictions,
            destination=destination,
        )
        
        assert result == destination
        assert Path(destination).exists()
        
        # Verify file contents
        df = pd.read_csv(destination)
        assert len(df) == 2
        assert 'feature_0' in df.columns
        assert 'target' in df.columns
        assert 'prediction' in df.columns

    def test_persist_reference_dataset_respects_max_rows(self, temp_dir: Path):
        """Test persist_reference_dataset respects max_rows."""
        features = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [5.0, 3.3, 1.4, 0.2]]
        targets = [0, 1, 0]
        
        destination = str(temp_dir / "reference.csv")
        
        result = persist_reference_dataset(
            features=features,
            targets=targets,
            destination=destination,
            max_rows=2,
        )
        
        assert result == destination
        
        # Verify only 2 rows saved
        df = pd.read_csv(destination)
        assert len(df) == 2

    def test_persist_reference_dataset_handles_missing_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """Test persist_reference_dataset handles missing env var."""
        monkeypatch.delenv("REFERENCE_DATASET_URI", raising=False)
        
        features = [[5.1, 3.5, 1.4, 0.2]]
        targets = [0]
        
        result = persist_reference_dataset(
            features=features,
            targets=targets,
            destination=None,
        )
        
        assert result is None

    def test_persist_reference_dataset_uses_env_var(self, monkeypatch: pytest.MonkeyPatch, temp_dir: Path):
        """Test persist_reference_dataset uses environment variable."""
        destination = str(temp_dir / "reference.csv")
        monkeypatch.setenv("REFERENCE_DATASET_URI", destination)
        
        features = [[5.1, 3.5, 1.4, 0.2]]
        targets = [0]
        
        result = persist_reference_dataset(
            features=features,
            targets=targets,
        )
        
        assert result == destination
        assert Path(destination).exists()

    def test_emit_prediction_log_creates_structured_log_entry(self, caplog: pytest.LogCaptureFixture):
        """Test emit_prediction_log creates structured log entry."""
        features = [[5.1, 3.5, 1.4, 0.2]]
        predictions = [0]
        
        event_id = emit_prediction_log(
            features=features,
            predictions=predictions,
        )
        
        assert event_id is not None
        assert len(event_id) > 0
        
        # Check that log was emitted
        assert len(caplog.records) > 0
        log_record = caplog.records[-1]
        assert log_record.levelname == "INFO"
        assert "Prediction log emitted" in log_record.message

    def test_emit_prediction_log_includes_metadata(self, caplog: pytest.LogCaptureFixture):
        """Test emit_prediction_log includes metadata."""
        features = [[5.1, 3.5, 1.4, 0.2]]
        predictions = [0]
        metadata = {"path": "/predict", "model": "test-model"}
        
        event_id = emit_prediction_log(
            features=features,
            predictions=predictions,
            metadata=metadata,
        )
        
        assert event_id is not None
        
        # Check metadata in log
        log_record = caplog.records[-1]
        assert hasattr(log_record, 'metadata') or 'metadata' in str(log_record.extra)

    def test_persist_reference_dataset_handles_file_uri(self, temp_dir: Path):
        """Test persist_reference_dataset handles file:// URI."""
        features = [[5.1, 3.5, 1.4, 0.2]]
        targets = [0]
        
        destination = f"file://{temp_dir / 'reference.csv'}"
        
        result = persist_reference_dataset(
            features=features,
            targets=targets,
            destination=destination,
        )
        
        assert result == destination
        # File should exist (without file:// prefix)
        assert Path(temp_dir / "reference.csv").exists()

