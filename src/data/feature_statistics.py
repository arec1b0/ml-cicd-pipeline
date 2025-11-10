"""
Feature statistics for input validation.

This module computes and caches statistics about training data features
to enable runtime validation of prediction inputs.

Comments and docstrings are written in English per repo standard.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Cache for feature statistics
_STATS_CACHE: Dict[str, Dict[str, Any]] = {}


def compute_feature_statistics(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for all numeric features in a dataset.
    
    This function reads a CSV file and computes min, max, mean, and std for each
    numeric feature (excluding the target column). Statistics are cached to avoid
    recomputation.
    
    Args:
        data_path: Path to the CSV file containing training data.
    
    Returns:
        Dictionary mapping feature names to dictionaries containing:
            - min: Minimum value in training data
            - max: Maximum value in training data
            - mean: Mean value in training data
            - std: Standard deviation in training data
            - type: Data type (float, int, etc.)
    
    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file cannot be parsed or contains no numeric features.
    """
    data_path_str = str(data_path)
    
    # Return cached statistics if available
    if data_path_str in _STATS_CACHE:
        logger.debug(f"Using cached feature statistics for {data_path_str}")
        return _STATS_CACHE[data_path_str]
    
    # Check if file exists
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(path)
        logger.info(f"Loaded data with shape {df.shape} from {data_path}")
        
        # Exclude target column (common names: target, label, y, class)
        target_names = {'target', 'label', 'y', 'class', 'class_label'}
        feature_cols = [col for col in df.columns if col.lower() not in target_names]
        
        # Compute statistics for each numeric feature
        stats: Dict[str, Dict[str, Any]] = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                stats[col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'median': float(col_data.median()),
                    'q1': float(col_data.quantile(0.25)),
                    'q3': float(col_data.quantile(0.75)),
                    'count': int(len(col_data)),
                    'type': str(col_data.dtype),
                }
                
                logger.debug(
                    f"Feature {col}: min={stats[col]['min']:.3f}, "
                    f"max={stats[col]['max']:.3f}, mean={stats[col]['mean']:.3f}"
                )
        
        if not stats:
            raise ValueError(f"No numeric features found in {data_path}")
        
        # Cache the statistics
        _STATS_CACHE[data_path_str] = stats
        logger.info(f"Computed statistics for {len(stats)} features")
        
        return stats
    
    except Exception as exc:
        logger.error(f"Failed to compute feature statistics: {exc}")
        raise


def get_feature_statistics(data_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get feature statistics, computing from data if necessary.
    
    If data_path is not provided, defaults to the training data in the project.
    
    Args:
        data_path: Optional path to CSV file. Defaults to data/processed/train.csv
    
    Returns:
        Dictionary mapping feature names to statistics dictionaries.
    """
    if data_path is None:
        data_path = "data/processed/train.csv"
    
    return compute_feature_statistics(data_path)


def validate_feature_range(
    values: list[float],
    feature_stats: Dict[str, Dict[str, Any]],
    strict_mode: bool = False
) -> tuple[bool, list[str]]:
    """Validate that feature values are within expected ranges.
    
    This function checks if each value is within the min/max range observed
    in the training data. In strict mode, it also checks for outliers using
    the interquartile range (IQR) method.
    
    Args:
        values: List of feature values to validate.
        feature_stats: Dictionary of feature statistics from compute_feature_statistics.
        strict_mode: If True, flag values outside IQR*1.5 as potential outliers.
    
    Returns:
        tuple: (is_valid, messages)
            - is_valid: True if all checks pass in non-strict mode
            - messages: List of warning/error messages for out-of-range or outlier values
    """
    if len(values) != len(feature_stats):
        return False, [f"Feature count mismatch: expected {len(feature_stats)}, got {len(values)}"]
    
    messages = []
    
    for i, (value, (feature_name, stats)) in enumerate(zip(values, feature_stats.items())):
        # Check for NaN and infinity
        if np.isnan(value) or np.isinf(value):
            messages.append(f"Feature {i} ({feature_name}): Invalid value {value} (NaN or infinity)")
            continue
        
        # Check range against training data
        min_val = stats['min']
        max_val = stats['max']
        
        if not (min_val <= value <= max_val):
            messages.append(
                f"Feature {i} ({feature_name}): Value {value:.3f} outside training range "
                f"[{min_val:.3f}, {max_val:.3f}]"
            )
        
        # Check for outliers using IQR method (strict mode)
        if strict_mode:
            q1 = stats['q1']
            q3 = stats['q3']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if not (lower_bound <= value <= upper_bound):
                messages.append(
                    f"Feature {i} ({feature_name}): Potential outlier {value:.3f} "
                    f"(outside IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}])"
                )
    
    # In non-strict mode, we only warn about out-of-range values
    # In strict mode, we also flag outliers
    is_valid = len(messages) == 0 or (not strict_mode and all("outlier" not in m for m in messages))
    
    return is_valid, messages


def clear_statistics_cache() -> None:
    """Clear the cached feature statistics.
    
    Useful for testing or when reloading training data.
    """
    global _STATS_CACHE
    _STATS_CACHE.clear()
    logger.info("Cleared feature statistics cache")


# Initialize statistics on module load
def _initialize_defaults():
    """Initialize default feature statistics from training data."""
    try:
        train_path = Path("data/processed/train.csv")
        if train_path.exists():
            compute_feature_statistics(str(train_path))
    except Exception as exc:
        logger.warning(f"Failed to initialize default statistics: {exc}")


# Initialize on import
_initialize_defaults()

