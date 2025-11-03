"""
Inference helper.
Minimal, single-responsibility loader + predictor for tests and microservice wiring.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np

class ModelWrapper:
    """
    Wrapper around scikit-learn models to provide a stable predict API.
    """
    def __init__(self, model: Any):
        self._model = model

    def predict(self, features: Sequence[Sequence[float]]) -> list:
        """
        Predict labels for batch of feature vectors.
        Args:
            features: 2D sequence of floats.
        Returns:
            list of predicted labels.
        """
        arr = np.asarray(features)
        preds = self._model.predict(arr)
        return preds.tolist()

def load_model(path: Path) -> ModelWrapper:
    """
    Load model from disk and wrap it.
    """
    model = joblib.load(path)
    return ModelWrapper(model)
