"""
Tests for data validators.
"""

import tempfile
from pathlib import Path

import pandas as pd
from src.data import validators


def test_validate_dataframe_ok():
    df = validators.load_sample_dataframe()
    ok, msg = validators.validate_dataframe(df)
    assert ok
    assert msg == "OK"


def test_validate_dataframe_missing_column():
    df = validators.load_sample_dataframe().drop(columns=["sepal_length"])
    ok, msg = validators.validate_dataframe(df)
    assert not ok
    assert "Missing required columns" in msg


def test_validate_dataframe_with_nan():
    df = validators.load_sample_dataframe()
    df.loc[0, "sepal_length"] = float("nan")
    ok, msg = validators.validate_dataframe(df)
    assert not ok
    assert "NaN values present" in msg


def test_validate_csv_file_ok():
    df = validators.load_sample_dataframe()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = Path(f.name)

    try:
        ok, msg = validators.validate_csv_file(temp_path)
        assert ok
        assert msg == "OK"
    finally:
        temp_path.unlink()


def test_validate_csv_file_missing():
    ok, msg = validators.validate_csv_file(Path("/nonexistent/file.csv"))
    assert not ok
    assert "File not found" in msg


def test_required_columns_defined():
    assert hasattr(validators, "REQUIRED_COLUMNS")
    assert len(validators.REQUIRED_COLUMNS) == 5
    assert "sepal_length" in validators.REQUIRED_COLUMNS
