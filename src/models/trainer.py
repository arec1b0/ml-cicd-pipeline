"""
Trainer module.
Contains a minimal, testable training function that follows SRP and is dependency-injectable.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@dataclass
class TrainResult:
    model_path: Path
    accuracy: float

def train(output_path: Path) -> TrainResult:
    """
    Train a simple RandomForest on Iris dataset and persist the model.
    Args:
        output_path: Path to write the trained model (file).
    Returns:
        TrainResult with saved path and validation accuracy.
    """
    data = load_iris()
    X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    # Save metrics alongside model for CI checks
    metrics_path = output_path.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf8") as fh:
        json.dump({"accuracy": acc}, fh)

    return TrainResult(model_path=output_path, accuracy=acc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a lightweight model (for CI).")
    parser.add_argument("--output", required=True, type=Path, help="Output model file path (pkl).")
    parser.add_argument("--metrics", required=False, type=Path, help="Output metrics JSON path.")
    args = parser.parse_args()

    result = train(args.output)
    if args.metrics:
        import json
        with open(args.metrics, "w", encoding="utf8") as mf:
            json.dump({"accuracy": result.accuracy}, mf)
    print(f"Model saved to: {result.model_path} with accuracy={result.accuracy:.4f}")
