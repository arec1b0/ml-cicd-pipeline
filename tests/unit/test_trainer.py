"""
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
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal lengt
# tests/unit/test_trainer.py
@'
"""
Tests for trainer module.
"""

from pathlib import Path
import tempfile
from src.models import trainer

def test_train_creates_model_and_metrics(tmp_path: Path):
    out_model = tmp_path / "model.pkl"
    result = trainer.train(out_model)
    assert out_model.exists()
    # metrics file should be created by trainer implementation
    metrics_file = out_model.parent / "metrics.json"
    assert metrics_file.exists()
    assert result.accuracy >= 0.0 and result.accuracy <= 1.0
