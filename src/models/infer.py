"""
Inference helper.
Minimal, single-responsibility loader + predictor for tests and microservice wiring.
Supports both ONNX and sklearn models, with ONNX preferred for production.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence, Optional
import logging
import joblib
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelWrapper:
    """
    Wrapper around ONNX or scikit-learn models to provide a stable predict API.
    """
    def __init__(self, model: Any, is_onnx: bool = False, input_name: Optional[str] = None):
        self._model = model
        self._is_onnx = is_onnx
        self._input_name = input_name or "float_input"

    def predict(self, features: Sequence[Sequence[float]]) -> list:
        """
        Predict labels for batch of feature vectors.
        Args:
            features: 2D sequence of floats.
        Returns:
            list of predicted labels.
        """
        arr = np.asarray(features, dtype=np.float32)
        
        if self._is_onnx:
            # ONNX inference
            outputs = self._model.run(None, {self._input_name: arr})
            # ONNX outputs are typically numpy arrays, get first output
            preds = outputs[0]
            # If output is 2D (e.g., probabilities), get argmax; if 1D, use directly
            if len(preds.shape) > 1:
                preds = np.argmax(preds, axis=1)
            return preds.tolist()
        else:
            # sklearn inference
            preds = self._model.predict(arr)
            return preds.tolist()

def load_model(path: Path) -> ModelWrapper:
    """
    Load model from disk and wrap it.
    Tries to load ONNX model first, falls back to sklearn .pkl if ONNX unavailable.
    
    Args:
        path: Path to model file. For ONNX, looks for .onnx extension or 
              tries model.onnx in same directory as .pkl path.
    Returns:
        ModelWrapper instance
    """
    path_obj = Path(path)
    
    # Try ONNX first if available
    if ONNX_AVAILABLE:
        # If path is .pkl, try to find .onnx in same directory
        if path_obj.suffix == '.pkl':
            onnx_path = path_obj.parent / "model.onnx"
            if not onnx_path.exists():
                # Try model_onnx directory (MLflow structure)
                onnx_path = path_obj.parent.parent / "model_onnx" / "model.onnx"
            if onnx_path.exists():
                try:
                    session = ort.InferenceSession(str(onnx_path))
                    input_name = session.get_inputs()[0].name
                    logger.info(f"Loaded ONNX model from {onnx_path}")
                    return ModelWrapper(session, is_onnx=True, input_name=input_name)
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model from {onnx_path}: {e}, falling back to sklearn")
        
        # If path is already .onnx
        elif path_obj.suffix == '.onnx':
            try:
                session = ort.InferenceSession(str(path_obj))
                input_name = session.get_inputs()[0].name
                logger.info(f"Loaded ONNX model from {path_obj}")
                return ModelWrapper(session, is_onnx=True, input_name=input_name)
            except Exception as e:
                logger.warning(f"Failed to load ONNX model from {path_obj}: {e}, falling back to sklearn")
    
    # Fallback to sklearn/joblib
    try:
        model = joblib.load(path_obj)
        logger.info(f"Loaded sklearn model from {path_obj}")
        return ModelWrapper(model, is_onnx=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path_obj}: {e}") from e
