import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

def test_processed_files_exist():
    """
    Ensure preprocessing created the expected files.
    This test is CI-friendly: it fails early if data not prepared.
    """
    assert (Path("data/processed/train.csv")).exists(), "train.csv missing"
    assert (Path("data/processed/val.csv")).exists(), "val.csv missing"

def test_train_schema():
    df = pd.read_csv("data/processed/train.csv")
    assert "target" in df.columns
    assert df.shape[1] == 5  # 4 features + target
