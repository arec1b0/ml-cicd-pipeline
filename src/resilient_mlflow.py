"""
Resilient MLflow client wrapper with retry logic and circuit breaker patterns.

This module provides a wrapper around MLflow operations to handle transient failures
and prevent cascading failures when MLflow is unavailable.
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, ParamSpec

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure threshold exceeded, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for the CircuitBreaker.

    Attributes:
        failure_threshold: The number of consecutive failures that must occur
                           before the circuit opens.
        success_threshold: The number of consecutive successes that must occur
                           in the half-open state before the circuit closes.
        timeout: The number of seconds to wait in the open state before
                 transitioning to the half-open state.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: int = 60


@dataclass
class RetryConfig:
    """Configuration for the retry mechanism.

    Attributes:
        max_attempts: The maximum number of times to retry a failed operation.
        backoff_factor: The multiplier for the exponential backoff between retries.
        initial_delay: The initial delay in seconds before the first retry.
        max_delay: The maximum delay in seconds between retries.
        retryable_exceptions: A tuple of exception classes that should be retried.
    """
    max_attempts: int = 5
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 30.0
    retryable_exceptions: tuple = (
        MlflowException,
        ConnectionError,
        TimeoutError,
    )


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initializes the CircuitBreaker.

        Args:
            config: The configuration for the circuit breaker.
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

    def call(self, func: Callable[[], T]) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute

        Returns:
            Function result

        Raises:
            Exception: If circuit is OPEN or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state", extra={})
            else:
                raise MlflowException(
                    f"Circuit breaker is OPEN. MLflow is unavailable. "
                    f"Last failure: {self.last_failure_time}"
                )

        try:
            result = func()
            self._on_success()
            return result
        except (MlflowException, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.config.timeout)

    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery", extra={})
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning("Circuit breaker reopened after failure in HALF_OPEN state", extra={})
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                "Circuit breaker OPENED after failures",
                extra={"failure_count": self.failure_count}
            )

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED", extra={})

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
        }


def retry_with_backoff(
    retry_config: RetryConfig,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """A decorator that adds retry logic with exponential backoff to a function.

    This decorator can be used to automatically retry a function that may fail
    due to transient errors. It can also be combined with a `CircuitBreaker`
    to prevent cascading failures.

    Args:
        retry_config: The configuration for the retry mechanism.
        circuit_breaker: An optional `CircuitBreaker` instance to use.

    Returns:
        A decorated function with retry logic.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            last_exception = None

            while attempt < retry_config.max_attempts:
                try:
                    if circuit_breaker:
                        return circuit_breaker.call(lambda: func(*args, **kwargs))
                    else:
                        return func(*args, **kwargs)

                except retry_config.retryable_exceptions as e:
                    attempt += 1
                    last_exception = e

                    if attempt >= retry_config.max_attempts:
                        logger.error(
                            "Failed after maximum attempts",
                            extra={"attempts": attempt, "function": func.__name__},
                            exc_info=True,
                        )
                        raise

                    delay = min(
                        retry_config.initial_delay
                        * (retry_config.backoff_factor ** (attempt - 1)),
                        retry_config.max_delay,
                    )

                    logger.warning(
                        "Retry attempt failed, retrying",
                        extra={
                            "attempt": attempt,
                            "max_attempts": retry_config.max_attempts,
                            "function": func.__name__,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "delay_seconds": delay
                        }
                    )
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry loop completed without exception")

        return wrapper

    return decorator


class ResilientMlflowClient:
    """
    Wrapper around MLflow client with retry logic and circuit breaker.

    This client automatically retries transient failures and protects against
    cascading failures when MLflow is unavailable.

    Example:
        client = ResilientMlflowClient(
            tracking_uri="http://mlflow:5000",
            retry_config=RetryConfig(max_attempts=5),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10)
        )

        with client.start_run() as run:
            client.log_param("param", "value")
            client.log_metric("metric", 0.95)
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize resilient MLflow client.

        Args:
            tracking_uri: MLflow tracking server URI
            retry_config: Retry configuration (uses defaults if None)
            circuit_breaker_config: Circuit breaker config (uses defaults if None)
        """
        self.tracking_uri = tracking_uri
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._client: Optional[MlflowClient] = None

    @property
    def client(self) -> MlflowClient:
        """Returns the underlying MLflow client, creating it if necessary."""
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    @retry_with_backoff(RetryConfig())
    def set_tracking_uri(self, uri: str) -> None:
        """Set MLflow tracking URI with retry."""
        mlflow.set_tracking_uri(uri)
        self.tracking_uri = uri
        self._client = None  # Reset client to use new URI

    @retry_with_backoff(RetryConfig())
    def set_experiment(self, experiment_name: str) -> str:
        """Set MLflow experiment with retry."""
        return mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, **kwargs: Any):
        """
        Start MLflow run with retry logic.

        Args:
            **kwargs: Arguments to pass to mlflow.start_run()

        Yields:
            MLflow run context
        """

        @retry_with_backoff(self.retry_config, self.circuit_breaker)
        def _start_run() -> Any:
            return mlflow.start_run(**kwargs)

        run = _start_run()
        try:
            yield run
        finally:
            try:
                mlflow.end_run()
            except (MlflowException, RuntimeError) as e:
                logger.warning("Error ending run", extra={"error": str(e), "error_type": type(e).__name__})

    @retry_with_backoff(RetryConfig())
    def log_param(self, key: str, value: Any) -> None:
        """Log parameter with retry."""
        mlflow.log_param(key, value)

    @retry_with_backoff(RetryConfig())
    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters with retry."""
        mlflow.log_params(params)

    @retry_with_backoff(RetryConfig())
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric with retry."""
        mlflow.log_metric(key, value, step=step)

    @retry_with_backoff(RetryConfig())
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics with retry."""
        mlflow.log_metrics(metrics, step=step)

    @retry_with_backoff(RetryConfig())
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact with retry."""
        mlflow.log_artifact(local_path, artifact_path)

    @retry_with_backoff(RetryConfig())
    def log_model(self, model: Any, artifact_path: str, **kwargs: Any) -> Any:
        """Log model with retry."""
        # Determine model flavor and use appropriate log function
        if hasattr(mlflow, "sklearn") and "sk_model" in kwargs:
            return mlflow.sklearn.log_model(
                sk_model=model, artifact_path=artifact_path, **kwargs
            )
        elif hasattr(mlflow, "onnx") and "onnx_model" in kwargs:
            return mlflow.onnx.log_model(
                onnx_model=model, artifact_path=artifact_path, **kwargs
            )
        else:
            # Generic model logging
            return mlflow.log_model(model, artifact_path, **kwargs)

    @retry_with_backoff(RetryConfig())
    def get_model_version(self, name: str, version: str) -> Any:
        """Get model version with retry."""
        return self.client.get_model_version(name, version)

    @retry_with_backoff(RetryConfig())
    def get_latest_versions(self, name: str, stages: Optional[list[str]] = None) -> list[Any]:
        """Get latest model versions with retry."""
        return self.client.get_latest_versions(name, stages=stages)

    @retry_with_backoff(RetryConfig())
    def download_artifacts(self, artifact_uri: str, dst_path: str) -> str:
        """Download artifacts with retry."""
        return mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=dst_path
        )

    @retry_with_backoff(RetryConfig())
    def search_runs(self, experiment_ids: list[str], **kwargs: Any) -> list[Any]:
        """Search runs with retry."""
        return self.client.search_runs(experiment_ids, **kwargs)

    @retry_with_backoff(RetryConfig())
    def get_run(self, run_id: str) -> Any:
        """Get run with retry."""
        return self.client.get_run(run_id)

    def get_circuit_breaker_state(self) -> dict[str, Any]:
        """Returns the current state of the circuit breaker.

        Returns:
            A dictionary containing the current state of the circuit breaker.
        """
        return self.circuit_breaker.get_state()

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self.circuit_breaker.reset()

    def health_check(self) -> dict:
        """Performs a health check of the MLflow server.

        This method attempts to connect to the MLflow server and get its
        version. It also returns the current state of the circuit breaker.

        Returns:
            A dictionary containing the health check status, and if
            successful, the MLflow server version.
        """
        try:
            # Try to get server version as a health check
            version = self.client.get_server_version()
            return {
                "status": "healthy",
                "server_version": version,
                "circuit_breaker": self.get_circuit_breaker_state(),
            }
        except (MlflowException, ConnectionError, TimeoutError, OSError) as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
                "circuit_breaker": self.get_circuit_breaker_state(),
            }


# Convenience function to create client from environment variables
def create_resilient_client_from_env() -> ResilientMlflowClient:
    """
    Create resilient MLflow client from configuration.

    Uses configuration from src.app.config for all settings.

    Returns:
        Configured ResilientMlflowClient
    """
    from src.app.config import (
        MLFLOW_TRACKING_URI,
        MLFLOW_RETRY_MAX_ATTEMPTS,
        MLFLOW_RETRY_BACKOFF_FACTOR,
        MLFLOW_CIRCUIT_BREAKER_THRESHOLD,
        MLFLOW_CIRCUIT_BREAKER_TIMEOUT,
    )

    tracking_uri = MLFLOW_TRACKING_URI

    retry_config = RetryConfig(
        max_attempts=MLFLOW_RETRY_MAX_ATTEMPTS,
        backoff_factor=MLFLOW_RETRY_BACKOFF_FACTOR,
    )

    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=MLFLOW_CIRCUIT_BREAKER_THRESHOLD,
        timeout=MLFLOW_CIRCUIT_BREAKER_TIMEOUT,
    )

    return ResilientMlflowClient(
        tracking_uri=tracking_uri,
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
    )
