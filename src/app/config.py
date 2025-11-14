"""
Configuration loader for the inference service.
Uses Pydantic BaseSettings for validation and type safety.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ModelSource(str, Enum):
    """Valid model source types."""

    MLFLOW = "mlflow"
    LOCAL = "local"


class AppConfig(BaseSettings):
    """Application configuration with validation.

    All configuration is loaded from environment variables with sensible defaults.
    Validation ensures MODEL_SOURCE-specific requirements are met.
    """

    # Model source configuration
    model_source: ModelSource = Field(
        default=ModelSource.MLFLOW,
        env="MODEL_SOURCE",
        description="Source of the model: 'mlflow' or 'local'",
    )

    # Base directory for model paths (configurable for different container structures)
    model_base_dir: Path = Field(
        default=Path("/app"),
        env="MODEL_BASE_DIR",
        description="Base directory for model paths",
    )

    # Model path configuration
    model_path: Optional[Path] = Field(
        default=None,
        env="MODEL_PATH",
        description="Path to local model file (required if MODEL_SOURCE=local)",
    )

    # Model cache directory
    model_cache_dir: Path = Field(
        default=Path("/var/cache/ml-model"),
        env="MODEL_CACHE_DIR",
        description="Directory to cache downloaded models",
    )

    # Model auto-refresh configuration
    model_auto_refresh_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        env="MODEL_AUTO_REFRESH_SECONDS",
        description="Interval in seconds for auto-refreshing the model (0 disables, max 3600)",
    )

    # Batch size limits
    max_batch_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        env="MAX_BATCH_SIZE",
        description="Maximum batch size for inference requests (1-10000)",
    )

    # Expected feature dimension (default for Iris dataset)
    expected_feature_dimension: int = Field(
        default=4,
        ge=1,
        env="EXPECTED_FEATURE_DIMENSION",
        description="Expected number of input features (default: 4 for Iris dataset)",
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (e.g., 'INFO', 'DEBUG')",
    )

    log_format: str = Field(
        default="json",
        env="LOG_FORMAT",
        description="Log format: 'json' or 'text'",
    )

    # Correlation ID configuration
    correlation_id_header: str = Field(
        default="X-Correlation-ID",
        env="CORRELATION_ID_HEADER",
        description="HTTP header for correlation IDs",
    )

    # OpenTelemetry configuration
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None,
        env="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="OTLP endpoint for exporting OpenTelemetry traces",
    )

    otel_service_name: str = Field(
        default="ml-cicd-pipeline",
        env="OTEL_SERVICE_NAME",
        description="Service name for OpenTelemetry tracing",
    )

    otel_resource_attributes: Optional[str] = Field(
        default=None,
        env="OTEL_RESOURCE_ATTRIBUTES",
        description="Resource attributes in 'key1=value1,key2=value2' format",
    )

    # MLflow configuration
    mlflow_model_name: Optional[str] = Field(
        default=None,
        env="MLFLOW_MODEL_NAME",
        description="Name of the model in MLflow (required if MODEL_SOURCE=mlflow)",
    )

    mlflow_model_stage: str = Field(
        default="Production",
        env="MLFLOW_MODEL_STAGE",
        description="Stage of the model in MLflow (e.g., 'Production', 'Staging')",
    )

    mlflow_model_version: Optional[str] = Field(
        default=None,
        env="MLFLOW_MODEL_VERSION",
        description="Version of the model in MLflow (optional, uses stage if not set)",
    )

    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        env="MLFLOW_TRACKING_URI",
        description="URI of the MLflow tracking server (required if MODEL_SOURCE=mlflow)",
    )

    mlflow_tracking_username: Optional[str] = Field(
        default=None,
        env="MLFLOW_TRACKING_USERNAME",
        description="Username for authenticated MLflow tracking",
    )

    mlflow_tracking_password: Optional[str] = Field(
        default=None,
        env="MLFLOW_TRACKING_PASSWORD",
        description="Password for authenticated MLflow tracking",
    )

    # MLflow retry configuration
    mlflow_retry_max_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        env="MLFLOW_RETRY_MAX_ATTEMPTS",
        description="Maximum retry attempts for MLflow operations (1-20)",
    )

    mlflow_retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        env="MLFLOW_RETRY_BACKOFF_FACTOR",
        description="Backoff multiplier for retries (1.0-10.0)",
    )

    # MLflow circuit breaker configuration
    mlflow_circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        env="MLFLOW_CIRCUIT_BREAKER_THRESHOLD",
        description="Failure threshold for circuit breaker (1-100)",
    )

    mlflow_circuit_breaker_timeout: int = Field(
        default=60,
        ge=1,
        le=3600,
        env="MLFLOW_CIRCUIT_BREAKER_TIMEOUT",
        description="Circuit breaker timeout in seconds (1-3600)",
    )

    # Admin API configuration
    admin_api_token: Optional[str] = Field(
        default=None,
        env="ADMIN_API_TOKEN",
        description="Admin token for accessing administrative endpoints (required for admin endpoints)",
    )

    admin_token_header: str = Field(
        default="X-Admin-Token",
        env="ADMIN_TOKEN_HEADER",
        description="HTTP header for admin token",
    )

    # Prediction cache configuration
    prediction_cache_enabled: bool = Field(
        default=True,
        env="PREDICTION_CACHE_ENABLED",
        description="Enable LRU cache for predictions",
    )

    prediction_cache_max_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        env="PREDICTION_CACHE_MAX_SIZE",
        description="Maximum number of entries in prediction cache (1-100000)",
    )

    prediction_cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        env="PREDICTION_CACHE_TTL_SECONDS",
        description="Time-to-live for cache entries in seconds (1-3600)",
    )

    # Prediction history configuration
    prediction_history_enabled: bool = Field(
        default=False,
        env="PREDICTION_HISTORY_ENABLED",
        description="Enable prediction history persistence to TimescaleDB",
    )

    prediction_history_database_url: Optional[str] = Field(
        default=None,
        env="PREDICTION_HISTORY_DATABASE_URL",
        description="TimescaleDB connection string (postgresql://user:pass@host/db)",
    )

    prediction_history_table_name: str = Field(
        default="prediction_history",
        env="PREDICTION_HISTORY_TABLE_NAME",
        description="Table name for prediction history",
    )

    @field_validator("model_path", mode="before")
    @classmethod
    def resolve_model_path(cls, v: Optional[str | Path], info) -> Optional[Path]:
        """Resolve model path relative to MODEL_BASE_DIR if not absolute."""
        if v is None:
            # If not set, use default path relative to base dir
            # Note: In Pydantic v2, we can't access other fields in field_validator
            # So we'll handle this in model_validator instead
            return None
        path = Path(v)
        if path.is_absolute():
            return path
        # For relative paths, we'll resolve in model_validator
        return path

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is either 'json' or 'text'."""
        if v not in ("json", "text"):
            raise ValueError("LOG_FORMAT must be 'json' or 'text'")
        return v

    @field_validator("model_source", mode="before")
    @classmethod
    def normalize_model_source(cls, v: str) -> str:
        """Normalize model source to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def resolve_relative_model_path(self) -> "AppConfig":
        """Resolve relative model paths after all fields are set."""
        if self.model_path is None:
            # Use default path relative to base dir
            self.model_path = self.model_base_dir / "model" / "model" / "model.pkl"
        elif not self.model_path.is_absolute():
            # Resolve relative path against base dir
            self.model_path = self.model_base_dir / self.model_path
        return self

    def validate_model_source_requirements(self) -> None:
        """Validate MODEL_SOURCE-specific requirements.

        Raises:
            ValueError: If required configuration is missing for the chosen MODEL_SOURCE.
        """
        if self.model_source == ModelSource.MLFLOW:
            if not self.mlflow_model_name:
                raise ValueError(
                    "MLFLOW_MODEL_NAME is required when MODEL_SOURCE=mlflow. "
                    "Please set the MLFLOW_MODEL_NAME environment variable."
                )
            if not self.mlflow_tracking_uri:
                raise ValueError(
                    "MLFLOW_TRACKING_URI is required when MODEL_SOURCE=mlflow. "
                    "Please set the MLFLOW_TRACKING_URI environment variable."
                )
        elif self.model_source == ModelSource.LOCAL:
            if not self.model_path:
                raise ValueError(
                    "MODEL_PATH is required when MODEL_SOURCE=local. "
                    "Please set the MODEL_PATH environment variable."
                )
            if not self.model_path.exists():
                raise ValueError(
                    f"MODEL_PATH '{self.model_path}' does not exist. "
                    "Please ensure the model file exists at the specified path."
                )

    def validate_config(self) -> None:
        """Validate configuration and fail fast with clear error messages.

        This method should be called at application startup to ensure
        all required configuration is present and valid.

        Raises:
            ValueError: If configuration validation fails.
        """
        try:
            self.validate_model_source_requirements()
        except ValueError as e:
            logger.error("Configuration validation failed: %s", str(e))
            raise

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True


# Create global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Returns:
        The global AppConfig instance.
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


# Backward compatibility: expose individual config values as module-level variables
# These will be populated after config is initialized
def _init_module_vars() -> None:
    """Initialize module-level variables from config for backward compatibility."""
    config = get_config()
    globals().update(
        {
            "MODEL_SOURCE": config.model_source.value,
            "MODEL_PATH": config.model_path,
            "MODEL_CACHE_DIR": config.model_cache_dir,
            "MODEL_AUTO_REFRESH_SECONDS": config.model_auto_refresh_seconds,
            "MAX_BATCH_SIZE": config.max_batch_size,
            "EXPECTED_FEATURE_DIMENSION": config.expected_feature_dimension,
            "LOG_LEVEL": config.log_level,
            "LOG_FORMAT": config.log_format,
            "CORRELATION_ID_HEADER": config.correlation_id_header,
            "OTEL_EXPORTER_OTLP_ENDPOINT": config.otel_exporter_otlp_endpoint,
            "OTEL_SERVICE_NAME": config.otel_service_name,
            "OTEL_RESOURCE_ATTRIBUTES": config.otel_resource_attributes,
            "MLFLOW_MODEL_NAME": config.mlflow_model_name,
            "MLFLOW_MODEL_STAGE": config.mlflow_model_stage,
            "MLFLOW_MODEL_VERSION": config.mlflow_model_version,
            "MLFLOW_TRACKING_URI": config.mlflow_tracking_uri,
            "MLFLOW_TRACKING_USERNAME": config.mlflow_tracking_username,
            "MLFLOW_TRACKING_PASSWORD": config.mlflow_tracking_password,
            "MLFLOW_RETRY_MAX_ATTEMPTS": config.mlflow_retry_max_attempts,
            "MLFLOW_RETRY_BACKOFF_FACTOR": config.mlflow_retry_backoff_factor,
            "MLFLOW_CIRCUIT_BREAKER_THRESHOLD": config.mlflow_circuit_breaker_threshold,
            "MLFLOW_CIRCUIT_BREAKER_TIMEOUT": config.mlflow_circuit_breaker_timeout,
            "ADMIN_API_TOKEN": config.admin_api_token,
            "ADMIN_TOKEN_HEADER": config.admin_token_header,
            "PREDICTION_CACHE_ENABLED": config.prediction_cache_enabled,
            "PREDICTION_CACHE_MAX_SIZE": config.prediction_cache_max_size,
            "PREDICTION_CACHE_TTL_SECONDS": config.prediction_cache_ttl_seconds,
            "PREDICTION_HISTORY_ENABLED": config.prediction_history_enabled,
            "PREDICTION_HISTORY_DATABASE_URL": config.prediction_history_database_url,
            "PREDICTION_HISTORY_TABLE_NAME": config.prediction_history_table_name,
        }
    )


# Initialize module-level variables
_init_module_vars()
