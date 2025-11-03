"""
Tests for inference module.
"""

from pathlib import Path
from src.models import trainer, infer
import tempfile

def test_infer_predict_shape(tmp_path: Path):
    model_path = tmp_path / "m.pkl"
    res = trainer.train(model_path)
    wrapper = infer.load_model(model_path)
    # pass one sample shaped according to iris features
    sample = [[5.1, 3.5, 1.4, 0.2]]
    preds = wrapper.predict(sample)
    assert isinstance(preds, list)
    assert len(preds) == 1
