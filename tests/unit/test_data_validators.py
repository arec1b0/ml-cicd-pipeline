"""
Tests for data validators.
"""

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
