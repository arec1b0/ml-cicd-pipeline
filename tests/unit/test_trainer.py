"""
Tests for trainer module.
"""

from pathlib import Path
import json
from src.models import trainer


def test_train_creates_model_and_metrics(tmp_path: Path):
    out_model = tmp_path / "model.pkl"
    metrics_file = tmp_path / "metrics.json"

    result = trainer.train(output_path=out_model, metrics_path=metrics_file)

    assert out_model.exists()
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf8") as handle:
        metrics = json.load(handle)

    assert "accuracy" in metrics
    assert result.accuracy == metrics["accuracy"]
    assert 0.0 <= result.accuracy <= 1.0


def test_train_returns_model_uri(tmp_path: Path):
    result = trainer.train(output_path=tmp_path / "model.pkl")
    assert result.model_uri
    # model may or may not have an explicit version depending on MLflow backend
    assert result.run_id
