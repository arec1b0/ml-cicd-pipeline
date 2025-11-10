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

def test_model_wrapper_get_input_dimension(tmp_path: Path):
    """Test that the ModelWrapper can extract the correct input dimension from the model."""
    model_path = tmp_path / "m.pkl"
    res = trainer.train(model_path)
    wrapper = infer.load_model(model_path)

    # The Iris dataset has 4 features
    input_dim = wrapper.get_input_dimension()
    assert input_dim is not None, "Input dimension should be extractable from model"
    assert input_dim == 4, f"Expected 4 features for Iris dataset, got {input_dim}"
