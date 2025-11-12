"""
Utilities for drift monitoring integration.
Provides helpers to persist reference datasets and emit lightweight
prediction event logs that downstream monitoring jobs can ingest.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


REFERENCE_DATASET_URI_ENV = "REFERENCE_DATASET_URI"
REFERENCE_DATASET_MAX_ROWS_ENV = "REFERENCE_DATASET_MAX_ROWS"

_logger = logging.getLogger("drift-monitoring")


def _normalise_destination(uri: str) -> str:
    """Normalizes a URI by removing the "file://" prefix if present.

    This is a helper function to ensure compatibility with libraries that
    expect a filesystem path instead of a file URI.

    Args:
        uri: The URI to normalize.

    Returns:
        The normalized URI as a string.
    """
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def load_dataset_from_uri(uri: str) -> pd.DataFrame:
    """Loads a dataset from a URI.

    This function supports local file paths and fsspec-compatible remote
    storage URIs (e.g., "s3://", "gs://").

    Args:
        uri: The URI of the dataset to load.

    Returns:
        A pandas DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified local file does not exist.
    """
    resolved = _normalise_destination(uri)
    if (
        resolved.startswith(("s3://", "gs://", "az://", "azure://"))
        or "://" in resolved
    ):
        return pd.read_csv(resolved)

    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def persist_reference_dataset(
    features: Sequence[Sequence[float]],
    targets: Sequence[float | int] | None,
    *,
    predictions: Sequence[float | int] | None = None,
    destination: str | None = None,
    max_rows: int | None = None,
) -> str | None:
    """Persists a reference dataset for drift monitoring.

    This function saves a dataset, typically a training or validation set,
    to a specified location. This dataset can then be used as a baseline
    for detecting data drift in production.

    Args:
        features: A 2D sequence of feature vectors.
        targets: A 1D sequence of target values.
        predictions: An optional 1D sequence of model predictions.
        destination: The URI to save the dataset to. If not provided, the
                     `REFERENCE_DATASET_URI` environment variable is used.
        max_rows: An optional maximum number of rows to save.

    Returns:
        The URI where the dataset was saved, or None if persistence is disabled.
    """
    destination = destination or os.getenv(REFERENCE_DATASET_URI_ENV)
    if not destination:
        _logger.debug(
            "Skipping reference dataset persistence; REFERENCE_DATASET_URI not set",
            extra={},
        )
        return None

    try:
        limit_env = os.getenv(REFERENCE_DATASET_MAX_ROWS_ENV)
        if max_rows is None and limit_env:
            max_rows = int(limit_env)
    except ValueError:
        _logger.warning(
            "Invalid REFERENCE_DATASET_MAX_ROWS value, falling back to unlimited rows",
            extra={"value": os.getenv(REFERENCE_DATASET_MAX_ROWS_ENV)},
        )
        max_rows = max_rows or None

    arr = np.asarray(features)
    if arr.ndim != 2:
        raise ValueError("features must be a 2D array-like structure")

    df = pd.DataFrame(arr, columns=[f"feature_{idx}" for idx in range(arr.shape[1])])
    if targets is not None:
        df["target"] = np.asarray(targets)
    if predictions is not None:
        df["prediction"] = np.asarray(predictions)

    if max_rows is not None:
        df = df.head(max_rows)

    resolved_destination = _normalise_destination(destination)
    if (
        resolved_destination.startswith(("s3://", "gs://", "az://", "azure://"))
        or "://" in resolved_destination
    ):
        # pandas uses fsspec under the hood for cloud URIs when the relevant
        # filesystem drivers are installed (e.g., s3fs, gcsfs).
        df.to_csv(resolved_destination, index=False)
    else:
        path = Path(resolved_destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    _logger.info(
        "Persisted reference dataset",
        extra={"destination": destination, "rows": len(df)},
    )
    return destination


def emit_prediction_log(
    *,
    features: Sequence[Sequence[float]],
    predictions: Sequence[float | int],
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Emits a JSON-formatted log entry for a prediction.

    This function is designed to be used as a background task to log
    prediction data. The structured JSON output is suitable for ingestion
    by log analysis tools like Loki.

    Args:
        features: The batch of feature vectors that were passed to the model.
        predictions: The batch of predictions that were returned by the model.
        metadata: Optional metadata to include in the log entry.

    Returns:
        The unique ID of the generated event.
    """
    event_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build structured log entry with all prediction data for Loki ingestion
    log_extra: dict[str, object] = {
        "event": "prediction",
        "event_id": event_id,
        "timestamp": timestamp,
        "features": features,
        "predictions": predictions,
    }
    if metadata:
        log_extra["metadata"] = dict(metadata)

    # Log with full payload in extra for structured JSON logging (Loki-compatible)
    _logger.info("Prediction log emitted", extra=log_extra)
    return event_id
