"""
Trainer module.
Contains a minimal, testable training function that follows SRP and is dependency-injectable.
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import mlflow
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.utils.drift import persist_reference_dataset

logger = logging.getLogger(__name__)

@dataclass
class TrainResult:
    accuracy: float
    run_id: str
    model_uri: str
    model_version: Optional[str] = None
    model_path: Optional[Path] = None
    reference_dataset_uri: Optional[str] = None


def _configure_mlflow() -> Tuple[str, str]:
    """
    Configure MLflow tracking and experiment from environment variables.
    Returns:
        Tuple of (tracking_uri, registered_model_name)
    Raises:
        RuntimeError if required env vars missing.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        # fall back to local file store for developer experience/tests
        tracking_uri = f"file://{(Path.cwd() / 'mlruns').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ml-cicd-pipeline")
    mlflow.set_experiment(experiment_name)

    registered_model_name = os.environ.get("MLFLOW_MODEL_NAME", "iris-random-forest")
    return tracking_uri, registered_model_name


def train(output_path: Optional[Path] = None, metrics_path: Optional[Path] = None) -> TrainResult:
    """
    Train a simple RandomForest on Iris dataset and log to MLflow.
    Args:
        output_path: Optional path to write the trained model (file).
        metrics_path: Optional path to write metrics JSON.
    Returns:
        TrainResult with MLflow metadata and optional saved path.
    """
    _, registered_model_name = _configure_mlflow()

    data = load_iris()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))

    reference_dataset_uri = None
    try:
        reference_dataset_uri = persist_reference_dataset(
            X_train, y_train, predictions=train_preds
        )
        if reference_dataset_uri:
            logger.info("Persisted training split to %s for drift monitoring.", reference_dataset_uri)
    except Exception as exc:
        logger.warning("Failed to persist reference dataset: %s", exc, exc_info=True)

    # Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        
        # Infer model signature for better tracking
        signature = infer_signature(X_val, preds)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_model_name
        )
        
        # Convert to ONNX and upload to MLflow as additional artifact
        try:
            # Define input type for ONNX conversion (iris has 4 features)
            initial_type = [('float_input', FloatTensorType([None, 4]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Log ONNX model as additional artifact in the same run
            # Use the same registered_model_name so it's part of the same model version
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model_onnx",
                registered_model_name=registered_model_name
            )
            logger.info("Successfully converted and uploaded ONNX model to MLflow")
        except Exception as exc:
            logger.warning("Failed to convert model to ONNX: %s", exc, exc_info=True)

    # Also save locally for backward compatibility and testing
    saved_model_path: Optional[Path] = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        saved_model_path = output_path

    # Save metrics alongside model for CI checks
    resolved_metrics_path: Optional[Path] = None
    if metrics_path:
        resolved_metrics_path = metrics_path
    elif output_path:
        resolved_metrics_path = output_path.parent / "metrics.json"

    if resolved_metrics_path:
        resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_metrics_path, "w", encoding="utf8") as fh:
            json.dump({"accuracy": acc}, fh)

    return TrainResult(
        accuracy=acc,
        run_id=run.info.run_id,
        model_uri=model_info.model_uri,
        model_version=getattr(model_info, "version", None),
        model_path=saved_model_path,
        reference_dataset_uri=reference_dataset_uri,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a lightweight model (for CI).")
    parser.add_argument("--output", type=Path, help="Optional output model file path (pkl).")
    parser.add_argument("--metrics", type=Path, help="Optional output metrics JSON path.")
    args = parser.parse_args()

    result = train(output_path=args.output, metrics_path=args.metrics)

    output_msg = f"MLflow model registered at {result.model_uri} (run_id={result.run_id}"
    if result.model_version:
        output_msg += f", version={result.model_version}"
    output_msg += ")"
    if result.model_path:
        output_msg += f"; local copy saved to: {result.model_path}"
    if result.reference_dataset_uri:
        output_msg += f"; reference dataset persisted to: {result.reference_dataset_uri}"
    print(output_msg)
    print(f"MODEL_URI={result.model_uri}")
