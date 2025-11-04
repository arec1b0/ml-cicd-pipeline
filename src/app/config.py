"""
Configuration loader for the inference service.
Reads configuration from environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path

def _get_int(name: str, default: int) -> int:
    """
    Read environment variable as int with fallback.
    """
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def get_env(name: str, default: str | None = None) -> str | None:
    """
    Read environment variable with fallback.
    """
    val = os.getenv(name, default)
    return val

DEFAULT_MODEL_PATH = "/app/model/model/model.pkl"
# Override MODEL_PATH if you keep legacy layouts or local copies.
MODEL_PATH = Path(get_env("MODEL_PATH", DEFAULT_MODEL_PATH))
MODEL_SOURCE = get_env("MODEL_SOURCE", "mlflow")
MODEL_CACHE_DIR = Path(get_env("MODEL_CACHE_DIR", "/var/cache/ml-model"))
MODEL_AUTO_REFRESH_SECONDS = _get_int("MODEL_AUTO_REFRESH_SECONDS", 0)
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
LOG_FORMAT = get_env("LOG_FORMAT", "json")
CORRELATION_ID_HEADER = get_env("CORRELATION_ID_HEADER", "X-Correlation-ID")
OTEL_EXPORTER_OTLP_ENDPOINT = get_env("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = get_env("OTEL_SERVICE_NAME", "ml-cicd-pipeline")
OTEL_RESOURCE_ATTRIBUTES = get_env("OTEL_RESOURCE_ATTRIBUTES")
MLFLOW_MODEL_NAME = get_env("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_STAGE = get_env("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_MODEL_VERSION = get_env("MLFLOW_MODEL_VERSION")
MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI")
ADMIN_API_TOKEN = get_env("ADMIN_API_TOKEN")
ADMIN_TOKEN_HEADER = get_env("ADMIN_TOKEN_HEADER", "X-Admin-Token")
