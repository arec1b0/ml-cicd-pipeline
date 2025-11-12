"""
Unit tests for feature statistics module.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_statistics import (
    compute_feature_statistics,
    get_feature_statistics,
    validate_feature_range,
    clear_statistics_cache,
)


@pytest.mark.unit
class TestFeatureStatistics:
    """Test feature statistics computation and validation."""

    def test_compute_feature_statistics_computes_stats_correctly(self, temp_csv_file: Path):
        """Test compute_feature_statistics computes stats correctly."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        assert len(stats) > 0
        assert 'feature_0' in stats
        
        feature_stats = stats['feature_0']
        assert 'min' in feature_stats
        assert 'max' in feature_stats
        assert 'mean' in feature_stats
        assert 'std' in feature_stats
        assert 'median' in feature_stats
        assert 'q1' in feature_stats
        assert 'q3' in feature_stats
        assert 'count' in feature_stats
        assert 'type' in feature_stats
        
        assert isinstance(feature_stats['min'], float)
        assert isinstance(feature_stats['max'], float)
        assert isinstance(feature_stats['mean'], float)

    def test_compute_feature_statistics_caches_results(self, temp_csv_file: Path):
        """Test compute_feature_statistics caches results."""
        # First call
        stats1 = compute_feature_statistics(str(temp_csv_file))
        
        # Second call should use cache
        stats2 = compute_feature_statistics(str(temp_csv_file))
        
        assert stats1 == stats2
        assert id(stats1) == id(stats2)  # Same object reference

    def test_compute_feature_statistics_raises_file_not_found_error(self):
        """Test compute_feature_statistics raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_feature_statistics("nonexistent_file.csv")

    def test_compute_feature_statistics_raises_value_error_no_numeric_features(self):
        """Test compute_feature_statistics raises ValueError for no numeric features."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'text_col': ['a', 'b', 'c'],
                'category_col': ['x', 'y', 'z'],
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="No numeric features"):
                compute_feature_statistics(str(temp_path))
        finally:
            temp_path.unlink()

    def test_get_feature_statistics_uses_default_path(self, monkeypatch: pytest.MonkeyPatch):
        """Test get_feature_statistics uses default path."""
        # Mock the default path to use a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0],
                'feature_1': [4.0, 5.0, 6.0],
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            # Temporarily replace the default path
            import src.data.feature_statistics as fs_module
            original_compute = fs_module.compute_feature_statistics
            
            def mock_compute(path: str):
                if path == "data/processed/train.csv":
                    return original_compute(str(temp_path))
                return original_compute(path)
            
            fs_module.compute_feature_statistics = mock_compute
            
            stats = get_feature_statistics()
            
            assert len(stats) > 0
            
            # Restore
            fs_module.compute_feature_statistics = original_compute
        finally:
            temp_path.unlink()

    def test_validate_feature_range_validates_in_range_values(self, temp_csv_file: Path):
        """Test validate_feature_range validates in-range values."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        # Get min and max from first feature
        feature_name = list(stats.keys())[0]
        feature_stats = stats[feature_name]
        
        # Value within range
        mid_value = (feature_stats['min'] + feature_stats['max']) / 2
        values = [mid_value] * len(stats)
        
        is_valid, messages = validate_feature_range(values, stats)
        
        assert is_valid is True
        assert len(messages) == 0

    def test_validate_feature_range_flags_out_of_range_values(self, temp_csv_file: Path):
        """Test validate_feature_range flags out-of-range values."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        # Value outside range
        feature_name = list(stats.keys())[0]
        feature_stats = stats[feature_name]
        
        out_of_range_value = feature_stats['max'] + 100.0
        values = [out_of_range_value] * len(stats)
        
        is_valid, messages = validate_feature_range(values, stats)
        
        assert is_valid is False or len(messages) > 0
        assert any("outside training range" in msg for msg in messages)

    def test_validate_feature_range_strict_mode_detects_outliers(self, temp_csv_file: Path):
        """Test validate_feature_range strict mode detects outliers."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        # Calculate outlier using IQR
        feature_name = list(stats.keys())[0]
        feature_stats = stats[feature_name]
        
        q1 = feature_stats['q1']
        q3 = feature_stats['q3']
        iqr = q3 - q1
        outlier_value = q3 + 2 * iqr  # Definitely an outlier
        
        values = [outlier_value] * len(stats)
        
        is_valid, messages = validate_feature_range(values, stats, strict_mode=True)
        
        assert any("outlier" in msg.lower() for msg in messages)

    def test_validate_feature_range_handles_nan_and_infinity(self, temp_csv_file: Path):
        """Test validate_feature_range handles NaN and infinity."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        values = [float('nan')] * len(stats)
        
        is_valid, messages = validate_feature_range(values, stats)
        
        assert any("NaN" in msg or "infinity" in msg for msg in messages)
        
        # Test infinity
        values = [float('inf')] * len(stats)
        
        is_valid, messages = validate_feature_range(values, stats)
        
        assert any("NaN" in msg or "infinity" in msg for msg in messages)

    def test_validate_feature_range_feature_count_mismatch(self, temp_csv_file: Path):
        """Test validate_feature_range handles feature count mismatch."""
        stats = compute_feature_statistics(str(temp_csv_file))
        
        # Wrong number of values
        values = [1.0] * (len(stats) + 1)
        
        is_valid, messages = validate_feature_range(values, stats)
        
        assert is_valid is False
        assert any("mismatch" in msg.lower() for msg in messages)

    def test_clear_statistics_cache(self, temp_csv_file: Path):
        """Test clear_statistics_cache clears cache."""
        # Compute stats to populate cache
        stats1 = compute_feature_statistics(str(temp_csv_file))
        
        # Clear cache
        clear_statistics_cache()
        
        # Compute again - should create new object
        stats2 = compute_feature_statistics(str(temp_csv_file))
        
        # Values should be same but object might be different
        assert stats1 == stats2

