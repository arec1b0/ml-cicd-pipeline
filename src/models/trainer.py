"""
Trainer module.
Contains a minimal, testable training function that follows SRP and is dependency-injectable.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple
import os
import joblib
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@dataclass
class TrainResult:
    accuracy: float
    run_id: str
    model_uri: str
    model_version: Optional[str] = None
    model_path: Optional[Path] = None


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
    X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))

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
    print(output_msg)
    print(f"MODEL_URI={result.model_uri}")
