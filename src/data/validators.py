"""
<<<<<<< HEAD
Simple data validation utilities for CI checks.
"""

from __future__ import annotations

from typing import List, Tuple
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
    Returns a tuple of (is_valid, message).
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
    """
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    column_mapping = {
=======
Data validators module.
Provides a programmatic API and small CLI for CI data checks.
"""

from __future__ import annotations
from typing import Tuple
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that dataframe contains required columns and no NaNs in these columns.
    Returns (is_valid, message).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    # simple NaN check in required columns
    if df[REQUIRED_COLUMNS].isnull().any().any():
        return False, "NaN values present in required columns"
    return True, "OK"

def load_sample_dataframe() -> pd.DataFrame:
    """
    Load the iris dataset as a pandas DataFrame for validation examples.
    """
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "target"})
    # ensure column names match our REQUIRED_COLUMNS naming
    df = df.rename(columns={
>>>>>>> 61c46b32f946dbaf77506e97a46a5133ab2ba0e1
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
<<<<<<< HEAD
    }
    df = df.rename(columns=column_mapping)
    return df[REQUIRED_COLUMNS]
=======
    })
    return df
>>>>>>> 61c46b32f946dbaf77506e97a46a5133ab2ba0e1
