"""
Configuration helpers for the drift monitoring service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_env_int(name: str, default: int | None = None) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {val!r}") from exc


@dataclass
class DriftSettings:
    """Strongly typed configuration for the drift monitoring service.

    This class encapsulates all the settings required to configure the drift
    monitoring service, including data source URIs, Loki connection details,
    and evaluation parameters.

    Attributes:
        reference_dataset_uri: The URI of the reference dataset (e.g., training data).
        current_dataset_uri: The URI of the current dataset (e.g., production data).
        loki_base_url: The base URL of the Loki service for querying logs.
        loki_query: The LogQL query to execute against Loki to fetch production data.
        loki_range_minutes: The time range in minutes for the Loki query.
        evaluation_interval_seconds: The interval in seconds between drift evaluations.
        min_rows: The minimum number of rows required in the current dataset to run an evaluation.
        max_rows: The maximum number of rows to sample from the current dataset.
        log_level: The logging level for the service.
    """
    reference_dataset_uri: str
    current_dataset_uri: Optional[str]
    loki_base_url: Optional[str]
    loki_query: Optional[str]
    loki_range_minutes: int
    loki_batch_size: int  # Number of log entries per batch
    loki_max_entries: Optional[int]  # Maximum total entries to fetch
    evaluation_interval_seconds: int
    min_rows: int
    max_rows: Optional[int]
    log_level: str

    @classmethod
    def from_env(cls) -> "DriftSettings":
        """Creates a DriftSettings instance from environment variables.

        This classmethod reads all the required and optional configuration
        values from environment variables and constructs a DriftSettings object.

        Returns:
            A configured DriftSettings instance.

        Raises:
            ValueError: If a required environment variable is not set.
        """
        reference_dataset_uri = os.getenv("REFERENCE_DATASET_URI")
        if not reference_dataset_uri:
            raise ValueError("REFERENCE_DATASET_URI env variable is required for drift monitoring service.")

        return cls(
            reference_dataset_uri=reference_dataset_uri,
            current_dataset_uri=os.getenv("CURRENT_DATASET_URI"),
            loki_base_url=os.getenv("LOKI_BASE_URL"),
            loki_query=os.getenv("LOKI_QUERY"),
            loki_range_minutes=_get_env_int("LOKI_RANGE_MINUTES", 60),
            loki_batch_size=_get_env_int("LOKI_BATCH_SIZE", 1000),
            loki_max_entries=_get_env_int("LOKI_MAX_ENTRIES"),
            evaluation_interval_seconds=_get_env_int("DRIFT_EVALUATION_INTERVAL_SECONDS", 300),
            min_rows=_get_env_int("DRIFT_MIN_ROWS", 50),
            max_rows=_get_env_int("DRIFT_MAX_ROWS"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        """Validates the configuration settings.

        This method checks that the settings are valid and consistent. For
        example, it ensures that at least one source for the current dataset
        is configured.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.current_dataset_uri and not self.loki_base_url:
            raise ValueError(
                "Either CURRENT_DATASET_URI or LOKI_BASE_URL must be provided to source current production data."
            )

