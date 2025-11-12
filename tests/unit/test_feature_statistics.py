"""
Unit tests for feature statistics module (src/data/feature_statistics.py).

Tests cover:
- Feature statistics computation
- Statistics caching
- Feature range validation (strict/non-strict modes)
- IQR-based outlier detection
- Cache clearing
- Error handling for missing files
- Edge cases (NaN, infinity, outliers)
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.data.feature_statistics import (
    compute_feature_statistics,
    get_feature_statistics,
    validate_feature_range,
    clear_statistics_cache,
    _STATS_CACHE,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with numeric features."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'feature_3': [0.5, 1.5, 2.5, 3.5, 4.5],
        'target': [0, 1, 0, 1, 0],
    })


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_dataframe):
    """Create a sample CSV file."""
    csv_path = Path(temp_data_dir) / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


def test_compute_feature_statistics_calculates_min_max(sample_csv_file):
    """Test that compute_feature_statistics calculates min and max values."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    assert 'feature_1' in stats
    assert stats['feature_1']['min'] == 1.0
    assert stats['feature_1']['max'] == 5.0

    assert 'feature_2' in stats
    assert stats['feature_2']['min'] == 10.0
    assert stats['feature_2']['max'] == 50.0


def test_compute_feature_statistics_calculates_mean_std(sample_csv_file):
    """Test that compute_feature_statistics calculates mean and std."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    assert 'mean' in stats['feature_1']
    assert 'std' in stats['feature_1']

    # Check approximate values
    assert abs(stats['feature_1']['mean'] - 3.0) < 0.01
    assert stats['feature_1']['std'] > 0


def test_compute_feature_statistics_calculates_quantiles(sample_csv_file):
    """Test that compute_feature_statistics calculates quantiles."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    assert 'q1' in stats['feature_1']
    assert 'median' in stats['feature_1']
    assert 'q3' in stats['feature_1']

    # Q1, median, Q3 for [1,2,3,4,5]
    assert stats['feature_1']['median'] == 3.0
    assert stats['feature_1']['q1'] == 2.0
    assert stats['feature_1']['q3'] == 4.0


def test_compute_feature_statistics_excludes_target_column(sample_csv_file):
    """Test that target column is excluded from statistics."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    # Target column should not be in statistics
    assert 'target' not in stats


def test_compute_feature_statistics_caches_results(sample_csv_file):
    """Test that statistics are cached after first computation."""
    clear_statistics_cache()

    # First call
    stats1 = compute_feature_statistics(sample_csv_file)

    # Second call should return cached results
    with patch('pandas.read_csv') as mock_read_csv:
        stats2 = compute_feature_statistics(sample_csv_file)

        # read_csv should not be called again
        mock_read_csv.assert_not_called()

    assert stats1 == stats2


def test_compute_feature_statistics_raises_for_missing_file():
    """Test that compute_feature_statistics raises FileNotFoundError."""
    clear_statistics_cache()

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        compute_feature_statistics("/nonexistent/file.csv")


def test_compute_feature_statistics_raises_for_no_numeric_features(temp_data_dir):
    """Test that error is raised when no numeric features are found."""
    clear_statistics_cache()

    # Create CSV with only non-numeric columns
    df = pd.DataFrame({
        'text_col': ['a', 'b', 'c'],
        'target': [0, 1, 0],
    })
    csv_path = Path(temp_data_dir) / "non_numeric.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No numeric features found"):
        compute_feature_statistics(str(csv_path))


def test_get_feature_statistics_uses_default_path():
    """Test that get_feature_statistics uses default path when not provided."""
    with patch('src.data.feature_statistics.compute_feature_statistics') as mock_compute:
        mock_compute.return_value = {}

        get_feature_statistics()

        # Should use default path
        mock_compute.assert_called_once_with("data/processed/train.csv")


def test_get_feature_statistics_uses_provided_path():
    """Test that get_feature_statistics uses provided path."""
    with patch('src.data.feature_statistics.compute_feature_statistics') as mock_compute:
        mock_compute.return_value = {}

        custom_path = "/custom/path/data.csv"
        get_feature_statistics(custom_path)

        mock_compute.assert_called_once_with(custom_path)


def test_validate_feature_range_accepts_valid_values():
    """Test that validate_feature_range accepts values within range."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    values = [5.0, 50.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is True
    assert len(messages) == 0


def test_validate_feature_range_rejects_out_of_range_values():
    """Test that validate_feature_range rejects out-of-range values."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    values = [15.0, 50.0]  # feature_1 is out of range
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) > 0
    assert "outside training range" in messages[0].lower()


def test_validate_feature_range_rejects_nan_values():
    """Test that validate_feature_range rejects NaN values."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    values = [np.nan, 50.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) > 0
    assert "nan" in messages[0].lower()


def test_validate_feature_range_rejects_infinity_values():
    """Test that validate_feature_range rejects infinity values."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    values = [np.inf, 50.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) > 0
    assert "infinity" in messages[0].lower()


def test_validate_feature_range_detects_outliers_in_strict_mode():
    """Test that validate_feature_range detects outliers in strict mode."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    # IQR = 75 - 25 = 50
    # Lower bound = 25 - 1.5*50 = -50
    # Upper bound = 75 + 1.5*50 = 150
    # Value 120 is within data range but outside IQR bounds

    values = [120.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=True)

    # In strict mode, outliers should be flagged
    assert len(messages) > 0
    assert "outlier" in messages[0].lower()


def test_validate_feature_range_ignores_outliers_in_non_strict_mode():
    """Test that validate_feature_range ignores outliers in non-strict mode."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    # Value within range but outside IQR bounds
    values = [95.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    # In non-strict mode, should be valid (within min/max)
    assert is_valid is True


def test_validate_feature_range_returns_false_for_count_mismatch():
    """Test that validate_feature_range rejects mismatched feature counts."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    # Provide only 1 value instead of 2
    values = [5.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) > 0
    assert "mismatch" in messages[0].lower()


def test_validate_feature_range_provides_detailed_messages():
    """Test that validate_feature_range provides detailed error messages."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
    }

    values = [15.0, 150.0]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) == 2  # One for each out-of-range feature

    # Check that messages contain feature names and values
    assert "feature_1" in messages[0]
    assert "15.0" in messages[0] or "15" in messages[0]
    assert "feature_2" in messages[1]


def test_clear_statistics_cache_empties_cache():
    """Test that clear_statistics_cache empties the cache."""
    # Add something to cache
    _STATS_CACHE['test_path'] = {'feature_1': {'min': 0.0, 'max': 10.0}}

    assert len(_STATS_CACHE) > 0

    clear_statistics_cache()

    assert len(_STATS_CACHE) == 0


def test_compute_feature_statistics_handles_mixed_types(temp_data_dir):
    """Test that compute_feature_statistics handles DataFrames with mixed types."""
    clear_statistics_cache()

    df = pd.DataFrame({
        'numeric_col': [1.0, 2.0, 3.0],
        'text_col': ['a', 'b', 'c'],
        'int_col': [10, 20, 30],
        'target': [0, 1, 0],
    })
    csv_path = Path(temp_data_dir) / "mixed_types.csv"
    df.to_csv(csv_path, index=False)

    stats = compute_feature_statistics(str(csv_path))

    # Should have statistics for numeric columns only
    assert 'numeric_col' in stats
    assert 'int_col' in stats
    assert 'text_col' not in stats
    assert 'target' not in stats


def test_compute_feature_statistics_handles_nan_in_data(temp_data_dir):
    """Test that compute_feature_statistics handles NaN values in data."""
    clear_statistics_cache()

    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'target': [0, 1, 0, 1, 0],
    })
    csv_path = Path(temp_data_dir) / "with_nan.csv"
    df.to_csv(csv_path, index=False)

    stats = compute_feature_statistics(str(csv_path))

    # Should compute statistics excluding NaN
    assert 'feature_1' in stats
    assert stats['feature_1']['count'] == 4  # Excluding NaN
    assert stats['feature_1']['min'] == 1.0
    assert stats['feature_1']['max'] == 5.0


def test_compute_feature_statistics_stores_count(sample_csv_file):
    """Test that compute_feature_statistics stores row count."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    assert 'count' in stats['feature_1']
    assert stats['feature_1']['count'] == 5


def test_compute_feature_statistics_stores_type(sample_csv_file):
    """Test that compute_feature_statistics stores data type."""
    clear_statistics_cache()

    stats = compute_feature_statistics(sample_csv_file)

    assert 'type' in stats['feature_1']
    assert 'float' in stats['feature_1']['type'].lower()


def test_validate_feature_range_multiple_errors():
    """Test that validate_feature_range reports multiple errors."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 10.0, 'q1': 2.5, 'q3': 7.5},
        'feature_2': {'min': 0.0, 'max': 100.0, 'q1': 25.0, 'q3': 75.0},
        'feature_3': {'min': -10.0, 'max': 10.0, 'q1': -5.0, 'q3': 5.0},
    }

    # Multiple issues: out of range, NaN, infinity
    values = [15.0, np.nan, np.inf]
    is_valid, messages = validate_feature_range(values, stats, strict_mode=False)

    assert is_valid is False
    assert len(messages) == 3  # One for each feature


def test_validate_feature_range_iqr_calculation():
    """Test that IQR-based outlier detection works correctly."""
    stats = {
        'feature_1': {'min': 0.0, 'max': 100.0, 'q1': 20.0, 'q3': 80.0},
    }

    # IQR = 80 - 20 = 60
    # Lower bound = 20 - 1.5*60 = -70
    # Upper bound = 80 + 1.5*60 = 170

    # Test value within IQR bounds
    values_valid = [50.0]
    is_valid, _ = validate_feature_range(values_valid, stats, strict_mode=True)
    assert is_valid is True

    # Test value outside IQR bounds but within min/max
    values_outlier = [95.0]
    is_valid, messages = validate_feature_range(values_outlier, stats, strict_mode=True)
    assert "outlier" in messages[0].lower()


def test_compute_feature_statistics_logs_debug_info(sample_csv_file):
    """Test that compute_feature_statistics logs debug information."""
    clear_statistics_cache()

    with patch('src.data.feature_statistics.logger') as mock_logger:
        stats = compute_feature_statistics(sample_csv_file)

        # Verify logging calls
        assert mock_logger.info.called
        assert mock_logger.debug.called
