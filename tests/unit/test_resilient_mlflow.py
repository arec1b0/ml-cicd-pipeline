"""
Tests for the resilient MLflow client with retry logic and circuit breaker.
"""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest
from mlflow.exceptions import MlflowException

from src.resilient_mlflow import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResilientMlflowClient,
    RetryConfig,
    retry_with_backoff,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker functionality."""

    def test_circuit_breaker_starts_closed(self):
        """Circuit breaker should start in CLOSED state."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Circuit breaker should open after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        # Simulate 3 failures
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_rejects_requests_when_open(self):
        """Circuit breaker should reject requests when OPEN."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60)
        cb = CircuitBreaker(config)

        # Trigger opening
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
        except Exception:
            pass

        # Should be open now
        assert cb.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(MlflowException, match="Circuit breaker is OPEN"):
            cb.call(lambda: "success")

    def test_circuit_breaker_transitions_to_half_open(self):
        """Circuit breaker should transition to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=1)
        cb = CircuitBreaker(config)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN and succeed
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_success_threshold(self):
        """Circuit breaker should close after success threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout=1
        )
        cb = CircuitBreaker(config)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
        except Exception:
            pass

        # Wait and transition to HALF_OPEN
        time.sleep(1.1)

        # First success
        cb.call(lambda: "success")
        assert cb.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        cb.call(lambda: "success")
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_reopens_on_failure_in_half_open(self):
        """Circuit breaker should reopen if failure occurs in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=1)
        cb = CircuitBreaker(config)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
        except Exception:
            pass

        # Wait and transition to HALF_OPEN
        time.sleep(1.1)

        # Failure in HALF_OPEN should reopen
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("another error")))
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_reset(self):
        """Manual reset should close circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("test error")))
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN

        # Manual reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_get_state(self):
        """get_state should return circuit breaker status."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        state = cb.get_state()

        assert "state" in state
        assert "failure_count" in state
        assert "success_count" in state
        assert state["state"] == "closed"


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    def test_retry_succeeds_on_first_attempt(self):
        """Function should succeed without retry if first attempt works."""
        mock_func = Mock(return_value="success")

        @retry_with_backoff(RetryConfig(max_attempts=3))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Function should retry and eventually succeed."""
        mock_func = Mock(
            side_effect=[
                MlflowException("error 1"),
                MlflowException("error 2"),
                "success",
            ]
        )

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausts_attempts(self):
        """Function should raise after max attempts exhausted."""
        mock_func = Mock(side_effect=MlflowException("persistent error"))

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def test_func():
            return mock_func()

        with pytest.raises(MlflowException, match="persistent error"):
            test_func()

        assert mock_func.call_count == 3

    def test_retry_respects_exponential_backoff(self):
        """Retry should use exponential backoff between attempts."""
        config = RetryConfig(
            max_attempts=3, initial_delay=0.1, backoff_factor=2.0, max_delay=1.0
        )

        call_times = []

        def failing_func():
            call_times.append(time.time())
            raise MlflowException("error")

        @retry_with_backoff(config)
        def test_func():
            return failing_func()

        with pytest.raises(MlflowException):
            test_func()

        # Verify exponential backoff
        assert len(call_times) == 3
        # First retry delay: ~0.1s
        assert call_times[1] - call_times[0] >= 0.1
        # Second retry delay: ~0.2s
        assert call_times[2] - call_times[1] >= 0.2

    def test_retry_respects_max_delay(self):
        """Retry delay should not exceed max_delay."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            backoff_factor=10.0,
            max_delay=0.5,
        )

        call_times = []

        def failing_func():
            call_times.append(time.time())
            raise MlflowException("error")

        @retry_with_backoff(config)
        def test_func():
            return failing_func()

        with pytest.raises(MlflowException):
            test_func()

        # All delays should be capped at max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay <= 0.6  # Allow small buffer for execution time

    def test_retry_only_retries_specified_exceptions(self):
        """Retry should only catch and retry specified exceptions."""
        mock_func = Mock(side_effect=ValueError("not retryable"))

        @retry_with_backoff(
            RetryConfig(max_attempts=3, retryable_exceptions=(MlflowException,))
        )
        def test_func():
            return mock_func()

        # ValueError should not be retried
        with pytest.raises(ValueError, match="not retryable"):
            test_func()

        assert mock_func.call_count == 1


class TestResilientMlflowClient:
    """Tests for ResilientMlflowClient."""

    def test_client_initialization(self):
        """Client should initialize with provided configuration."""
        retry_config = RetryConfig(max_attempts=10)
        cb_config = CircuitBreakerConfig(failure_threshold=10)

        client = ResilientMlflowClient(
            tracking_uri="http://test:5000",
            retry_config=retry_config,
            circuit_breaker_config=cb_config,
        )

        assert client.tracking_uri == "http://test:5000"
        assert client.retry_config.max_attempts == 10
        assert client.circuit_breaker.config.failure_threshold == 10

    @patch("src.resilient_mlflow.mlflow")
    def test_set_experiment_with_retry(self, mock_mlflow):
        """set_experiment should use retry logic."""
        mock_mlflow.set_experiment.side_effect = [
            MlflowException("connection error"),
            "experiment_id",
        ]

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )

        result = client.set_experiment("test-experiment")

        assert result == "experiment_id"
        assert mock_mlflow.set_experiment.call_count == 2

    @patch("src.resilient_mlflow.mlflow")
    def test_log_params_with_retry(self, mock_mlflow):
        """log_params should use retry logic."""
        mock_mlflow.log_params.side_effect = [
            ConnectionError("network error"),
            None,
        ]

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )

        client.log_params({"param1": "value1"})

        assert mock_mlflow.log_params.call_count == 2

    @patch("src.resilient_mlflow.mlflow")
    def test_log_metric_with_retry(self, mock_mlflow):
        """log_metric should use retry logic."""
        mock_mlflow.log_metric.side_effect = [TimeoutError("timeout"), None]

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )

        client.log_metric("accuracy", 0.95)

        assert mock_mlflow.log_metric.call_count == 2

    @patch("src.resilient_mlflow.mlflow")
    def test_start_run_with_circuit_breaker(self, mock_mlflow):
        """start_run should use circuit breaker protection."""
        # Configure mock to fail repeatedly
        mock_mlflow.start_run.side_effect = MlflowException("persistent error")

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
        )

        # First attempt - should fail and open circuit
        with pytest.raises(MlflowException):
            with client.start_run():
                pass

        # Circuit should be open now
        assert client.circuit_breaker.state == CircuitState.OPEN

        # Second attempt should be rejected by circuit breaker
        with pytest.raises(MlflowException, match="Circuit breaker is OPEN"):
            with client.start_run():
                pass

    def test_get_circuit_breaker_state(self):
        """get_circuit_breaker_state should return current state."""
        client = ResilientMlflowClient()
        state = client.get_circuit_breaker_state()

        assert "state" in state
        assert state["state"] == "closed"

    def test_reset_circuit_breaker(self):
        """reset_circuit_breaker should reset to CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        client = ResilientMlflowClient(circuit_breaker_config=config)

        # Open circuit
        client.circuit_breaker._on_failure()
        assert client.circuit_breaker.state == CircuitState.OPEN

        # Reset
        client.reset_circuit_breaker()
        assert client.circuit_breaker.state == CircuitState.CLOSED

    @patch("src.resilient_mlflow.MlflowClient")
    def test_health_check_healthy(self, mock_mlflow_client):
        """health_check should return healthy status when working."""
        mock_client_instance = Mock()
        mock_client_instance.get_server_version.return_value = "2.7.0"
        mock_mlflow_client.return_value = mock_client_instance

        client = ResilientMlflowClient()
        health = client.health_check()

        assert health["status"] == "healthy"
        assert health["server_version"] == "2.7.0"
        assert "circuit_breaker" in health

    @patch("src.resilient_mlflow.MlflowClient")
    def test_health_check_unhealthy(self, mock_mlflow_client):
        """health_check should return unhealthy status on error."""
        mock_client_instance = Mock()
        mock_client_instance.get_server_version.side_effect = ConnectionError(
            "cannot connect"
        )
        mock_mlflow_client.return_value = mock_client_instance

        client = ResilientMlflowClient()
        health = client.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "circuit_breaker" in health

    @patch("src.resilient_mlflow.MlflowClient")
    def test_get_model_version_with_retry(self, mock_mlflow_client):
        """get_model_version should use retry logic."""
        mock_client_instance = Mock()
        mock_client_instance.get_model_version.side_effect = [
            ConnectionError("network error"),
            Mock(name="iris-model", version="1"),
        ]
        mock_mlflow_client.return_value = mock_client_instance

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )

        result = client.get_model_version("iris-model", "1")

        assert result.version == "1"
        assert mock_client_instance.get_model_version.call_count == 2

    @patch("src.resilient_mlflow.MlflowClient")
    def test_get_latest_versions_with_retry(self, mock_mlflow_client):
        """get_latest_versions should use retry logic."""
        mock_client_instance = Mock()
        mock_version = Mock(version="2", current_stage="Production")
        mock_client_instance.get_latest_versions.side_effect = [
            TimeoutError("timeout"),
            [mock_version],
        ]
        mock_mlflow_client.return_value = mock_client_instance

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01)
        )

        result = client.get_latest_versions("iris-model", stages=["Production"])

        assert len(result) == 1
        assert result[0].version == "2"
        assert mock_client_instance.get_latest_versions.call_count == 2


class TestResilientClientIntegration:
    """Integration tests for resilient client behavior."""

    @patch("src.resilient_mlflow.mlflow")
    def test_full_training_workflow_with_transient_failure(self, mock_mlflow):
        """Simulate full training workflow with transient MLflow failure."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # log_params succeeds
        mock_mlflow.log_params.return_value = None

        # log_metric fails once then succeeds
        mock_mlflow.log_metric.side_effect = [
            ConnectionError("network blip"),
            None,
        ]

        # Create client with retry
        client = ResilientMlflowClient(
            tracking_uri="http://test:5000",
            retry_config=RetryConfig(max_attempts=3, initial_delay=0.01),
        )

        # Execute training workflow
        client.set_experiment("test-experiment")

        with client.start_run() as run:
            client.log_params({"n_estimators": 10, "random_state": 42})
            client.log_metric("accuracy", 0.95)  # This will retry

        # Verify workflow completed successfully
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.log_params.called
        assert mock_mlflow.log_metric.call_count == 2  # 1 failure + 1 success

    @patch("src.resilient_mlflow.mlflow")
    def test_circuit_breaker_protects_during_outage(self, mock_mlflow):
        """Simulate MLflow outage with circuit breaker protection."""
        # Setup persistent failure
        mock_mlflow.start_run.side_effect = MlflowException("MLflow down")

        client = ResilientMlflowClient(
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, timeout=60
            ),
        )

        # First call - exhausts retries and opens circuit
        with pytest.raises(MlflowException):
            with client.start_run():
                pass

        # Verify circuit opened
        assert client.circuit_breaker.state == CircuitState.OPEN

        # Second call - immediately rejected by circuit breaker
        with pytest.raises(MlflowException, match="Circuit breaker is OPEN"):
            with client.start_run():
                pass

        # Should not have retried (circuit breaker blocked it)
        # First call made 2 attempts (max_attempts), second call was blocked
        assert mock_mlflow.start_run.call_count == 2


def test_create_resilient_client_from_env():
    """Test creating client from environment variables."""
    import os

    with patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://test:5000",
            "MLFLOW_RETRY_MAX_ATTEMPTS": "10",
            "MLFLOW_RETRY_BACKOFF_FACTOR": "3.0",
            "MLFLOW_CIRCUIT_BREAKER_THRESHOLD": "15",
            "MLFLOW_CIRCUIT_BREAKER_TIMEOUT": "120",
        },
    ):
        from src.resilient_mlflow import create_resilient_client_from_env

        client = create_resilient_client_from_env()

        assert client.tracking_uri == "http://test:5000"
        assert client.retry_config.max_attempts == 10
        assert client.retry_config.backoff_factor == 3.0
        assert client.circuit_breaker.config.failure_threshold == 15
        assert client.circuit_breaker.config.timeout == 120
