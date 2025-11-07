"""
Configuration loader for the inference service.
Reads configuration from environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path

def _get_int(name: str, default: int, max_value: int | None = None) -> int:
    """
    Read environment variable as int with fallback and optional max value validation.

    Args:
        name: The name of the environment variable.
        default: The default value to use if the environment variable is not set.
        max_value: The maximum allowed value (inclusive). If exceeded, raises ValueError.

    Returns:
        The value of the environment variable as an integer.

    Raises:
        ValueError: If the value exceeds max_value.
    """
    raw = get_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Security: {name}={value} exceeds maximum allowed value of {max_value}. "
                f"This could lead to resource exhaustion."
            )
        return value
    except ValueError:
        return default

def get_env(name: str, default: str | None = None) -> str | None:
    """Reads an environment variable with a fallback to a default value.

    Args:
        name: The name of the environment variable.
        default: The default value to use if the environment variable is not set.

    Returns:
        The value of the environment variable, or the default value.
    """
    val = os.getenv(name, default)
    return val

# Default path to the machine learning model file.
DEFAULT_MODEL_PATH = "/app/model/model/model.pkl"

# Path to the machine learning model file. Can be overridden via the MODEL_PATH environment variable.
MODEL_PATH = Path(get_env("MODEL_PATH", DEFAULT_MODEL_PATH))
# Source of the model, e.g., "mlflow" or "local".
MODEL_SOURCE = get_env("MODEL_SOURCE", "mlflow")

# Directory to cache downloaded models.
MODEL_CACHE_DIR = Path(get_env("MODEL_CACHE_DIR", "/var/cache/ml-model"))

# Interval in seconds for auto-refreshing the model. 0 disables it.
# Maximum value: 3600 seconds (1 hour) to prevent resource exhaustion
MODEL_AUTO_REFRESH_SECONDS = _get_int("MODEL_AUTO_REFRESH_SECONDS", 0, max_value=3600)

# Maximum batch size for inference requests to prevent DoS attacks
# Maximum value: 10000 to prevent resource exhaustion
MAX_BATCH_SIZE = _get_int("MAX_BATCH_SIZE", 1000, max_value=10000)

# Logging level for the application (e.g., "INFO", "DEBUG").
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")

# Log format to use ("json" or "text").
LOG_FORMAT = get_env("LOG_FORMAT", "json")

# HTTP header used to track correlation IDs for requests.
CORRELATION_ID_HEADER = get_env("CORRELATION_ID_HEADER", "X-Correlation-ID")

# OTLP endpoint for exporting OpenTelemetry traces.
OTEL_EXPORTER_OTLP_ENDPOINT = get_env("OTEL_EXPORTER_OTLP_ENDPOINT")

# Service name for OpenTelemetry tracing.
OTEL_SERVICE_NAME = get_env("OTEL_SERVICE_NAME", "ml-cicd-pipeline")

# Resource attributes for OpenTelemetry tracing, in "key1=value1,key2=value2" format.
OTEL_RESOURCE_ATTRIBUTES = get_env("OTEL_RESOURCE_ATTRIBUTES")

# Name of the model in MLflow.
MLFLOW_MODEL_NAME = get_env("MLFLOW_MODEL_NAME")

# Stage of the model in MLflow (e.g., "Production", "Staging").
MLFLOW_MODEL_STAGE = get_env("MLFLOW_MODEL_STAGE", "Production")

# Version of the model in MLflow.
MLFLOW_MODEL_VERSION = get_env("MLFLOW_MODEL_VERSION")

# URI of the MLflow tracking server.
MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI")

# Admin token for accessing administrative endpoints.
ADMIN_API_TOKEN = get_env("ADMIN_API_TOKEN")

# HTTP header for the admin token.
ADMIN_TOKEN_HEADER = get_env("ADMIN_TOKEN_HEADER", "X-Admin-Token")
