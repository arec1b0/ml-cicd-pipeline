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
    """
    Strongly typed configuration for the drift monitoring service.
    """

    reference_dataset_uri: str
    current_dataset_uri: Optional[str]
    loki_base_url: Optional[str]
    loki_query: Optional[str]
    loki_range_minutes: int
    evaluation_interval_seconds: int
    min_rows: int
    max_rows: Optional[int]
    log_level: str

    @classmethod
    def from_env(cls) -> "DriftSettings":
        reference_dataset_uri = os.getenv("REFERENCE_DATASET_URI")
        if not reference_dataset_uri:
            raise ValueError("REFERENCE_DATASET_URI env variable is required for drift monitoring service.")

        return cls(
            reference_dataset_uri=reference_dataset_uri,
            current_dataset_uri=os.getenv("CURRENT_DATASET_URI"),
            loki_base_url=os.getenv("LOKI_BASE_URL"),
            loki_query=os.getenv("LOKI_QUERY"),
            loki_range_minutes=_get_env_int("LOKI_RANGE_MINUTES", 60),
            evaluation_interval_seconds=_get_env_int("DRIFT_EVALUATION_INTERVAL_SECONDS", 300),
            min_rows=_get_env_int("DRIFT_MIN_ROWS", 50),
            max_rows=_get_env_int("DRIFT_MAX_ROWS"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        """
        Ensure the settings contain a current data source definition.
        """
        if not self.current_dataset_uri and not self.loki_base_url:
            raise ValueError(
                "Either CURRENT_DATASET_URI or LOKI_BASE_URL must be provided to source current production data."
            )

