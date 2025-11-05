"""
Data validators module.
Provides a programmatic API and CLI for CI data checks.
"""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target",
]


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate a dataframe against the required schema.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, message)
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    if df[REQUIRED_COLUMNS].isnull().any().any():
        return False, "NaN values present in required columns"

    return True, "OK"


def load_sample_dataframe() -> pd.DataFrame:
    """
    Load the iris dataset as a pandas DataFrame with canonical column names.

    Returns:
        DataFrame with standardized column names
    """
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=column_mapping)
    return df[REQUIRED_COLUMNS]


def validate_csv_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate a CSV file against the required schema.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        df = pd.read_csv(file_path)
        return validate_dataframe(df)
    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except pd.errors.EmptyDataError:
        return False, f"Empty file: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


__all__ = [
    "REQUIRED_COLUMNS",
    "validate_dataframe",
    "load_sample_dataframe",
    "validate_csv_file",
]
