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
