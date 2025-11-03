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
    """
    Convert file:// URIs to filesystem paths for pandas compatibility.
    """
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def load_dataset_from_uri(uri: str) -> pd.DataFrame:
    """
    Load a dataset from a URI supporting local paths and fsspec-compatible schemes.
    Args:
        uri: Location of the dataset (CSV format).
    Returns:
        pandas DataFrame.
    Raises:
        FileNotFoundError when the local file is absent.
    """
    resolved = _normalise_destination(uri)
    if resolved.startswith(("s3://", "gs://", "az://", "azure://")) or "://" in resolved:
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
    """
    Persist the training/reference split so downstream jobs can detect drift.
    Args:
        features: 2D array-like of feature values.
        targets: 1D array-like with labels. Optional if unavailable.
        predictions: 1D array-like with model predictions. Optional.
        destination: Optional override for REFERENCE_DATASET_URI env.
        max_rows: Optional cap on number of rows to store.
    Returns:
        Location the dataset was written to, or None when disabled.
    """
    destination = destination or os.getenv(REFERENCE_DATASET_URI_ENV)
    if not destination:
        _logger.debug("Skipping reference dataset persistence; REFERENCE_DATASET_URI not set.")
        return None

    try:
        limit_env = os.getenv(REFERENCE_DATASET_MAX_ROWS_ENV)
        if max_rows is None and limit_env:
            max_rows = int(limit_env)
    except ValueError:
        _logger.warning(
            "Invalid REFERENCE_DATASET_MAX_ROWS value: %s. Falling back to unlimited rows.",
            os.getenv(REFERENCE_DATASET_MAX_ROWS_ENV),
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
    if resolved_destination.startswith(("s3://", "gs://", "az://", "azure://")) or "://" in resolved_destination:
        # pandas uses fsspec under the hood for cloud URIs when the relevant
        # filesystem drivers are installed (e.g., s3fs, gcsfs).
        df.to_csv(resolved_destination, index=False)
    else:
        path = Path(resolved_destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    _logger.info("Persisted reference dataset to %s (rows=%s)", destination, len(df))
    return destination


def emit_prediction_log(
    *,
    features: Sequence[Sequence[float]],
    predictions: Sequence[float | int],
    metadata: Mapping[str, object] | None = None,
) -> str:
    """
    Emit a JSON log line capturing the prediction request/response.
    Structured logs can be scraped by Loki/Promtail or other log forwarders.
    Args:
        features: Batch of features passed to the model.
        predictions: Batch of predictions returned by the model.
        metadata: Optional metadata such as request ID or model info.
    Returns:
        Generated event ID to assist correlating logs downstream.
    """
    event_id = str(uuid4())
    payload: dict[str, object] = {
        "event": "prediction",
        "event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "predictions": predictions,
    }
    if metadata:
        payload["metadata"] = dict(metadata)

    _logger.info(json.dumps(payload, default=str))
    return event_id
