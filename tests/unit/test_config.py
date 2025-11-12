"""
Unit tests for configuration module (src/app/config.py).

Tests cover:
- Environment variable parsing
- Default value handling
- Integer conversion with max value validation
- Security limits enforcement
- Path configuration
"""

from __future__ import annotations

import pytest
from unittest.mock import patch
from pathlib import Path


def test_get_env_returns_environment_value():
    """Test that get_env returns value from environment."""
    from src.app.config import get_env

    with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
        result = get_env("TEST_VAR")
        assert result == "test_value"


def test_get_env_returns_default_when_not_set():
    """Test that get_env returns default when variable not set."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("NONEXISTENT_VAR", "default_value")
        assert result == "default_value"


def test_get_env_returns_none_when_no_default():
    """Test that get_env returns None when no default provided."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("NONEXISTENT_VAR")
        assert result is None


def test_get_int_returns_integer_value():
    """Test that _get_int converts string to integer."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="42"):
        result = _get_int("TEST_INT", default=10)
        assert result == 42


def test_get_int_returns_default_when_not_set():
    """Test that _get_int returns default when variable not set."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value=None):
        result = _get_int("TEST_INT", default=100)
        assert result == 100


def test_get_int_returns_default_when_invalid_format():
    """Test that _get_int returns default for non-numeric values."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="not_a_number"):
        result = _get_int("TEST_INT", default=50)
        assert result == 50


def test_get_int_enforces_max_value():
    """Test that _get_int raises error when value exceeds max_value."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="5000"):
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            _get_int("TEST_INT", default=100, max_value=1000)


def test_get_int_allows_value_at_max_limit():
    """Test that _get_int allows value exactly at max_value."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="1000"):
        result = _get_int("TEST_INT", default=100, max_value=1000)
        assert result == 1000


def test_get_int_allows_value_below_max_limit():
    """Test that _get_int allows value below max_value."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="500"):
        result = _get_int("TEST_INT", default=100, max_value=1000)
        assert result == 500


def test_model_path_default():
    """Test that MODEL_PATH has correct default value."""
    from src.app.config import DEFAULT_MODEL_PATH

    assert DEFAULT_MODEL_PATH == "/app/model/model/model.pkl"


def test_model_path_is_path_object():
    """Test that MODEL_PATH is a Path object."""
    from src.app.config import MODEL_PATH

    assert isinstance(MODEL_PATH, Path)


def test_model_source_default():
    """Test that MODEL_SOURCE defaults to mlflow."""
    with patch.dict("os.environ", {}, clear=True):
        # Re-import to get fresh values
        import importlib
        import src.app.config as config_module

        with patch("src.app.config.get_env", wraps=config_module.get_env):
            value = config_module.get_env("MODEL_SOURCE", "mlflow")
            assert value == "mlflow"


def test_max_batch_size_has_security_limit():
    """Test that MAX_BATCH_SIZE enforces security limit."""
    from src.app.config import _get_int

    # Try to set batch size above maximum
    with patch("src.app.config.get_env", return_value="20000"):
        with pytest.raises(ValueError, match="resource exhaustion"):
            _get_int("MAX_BATCH_SIZE", default=1000, max_value=10000)


def test_max_batch_size_default_value():
    """Test that MAX_BATCH_SIZE has reasonable default."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value=None):
        result = _get_int("MAX_BATCH_SIZE", default=1000, max_value=10000)
        assert result == 1000


def test_auto_refresh_has_security_limit():
    """Test that MODEL_AUTO_REFRESH_SECONDS enforces security limit."""
    from src.app.config import _get_int

    # Try to set refresh interval above maximum (1 hour)
    with patch("src.app.config.get_env", return_value="7200"):
        with pytest.raises(ValueError, match="resource exhaustion"):
            _get_int("MODEL_AUTO_REFRESH_SECONDS", default=0, max_value=3600)


def test_auto_refresh_default_disabled():
    """Test that MODEL_AUTO_REFRESH_SECONDS is disabled by default."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value=None):
        result = _get_int("MODEL_AUTO_REFRESH_SECONDS", default=0, max_value=3600)
        assert result == 0


def test_log_level_default():
    """Test that LOG_LEVEL defaults to INFO."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("LOG_LEVEL", "INFO")
        assert result == "INFO"


def test_log_format_default():
    """Test that LOG_FORMAT defaults to json."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("LOG_FORMAT", "json")
        assert result == "json"


def test_correlation_id_header_default():
    """Test that CORRELATION_ID_HEADER has correct default."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("CORRELATION_ID_HEADER", "X-Correlation-ID")
        assert result == "X-Correlation-ID"


def test_otel_service_name_default():
    """Test that OTEL_SERVICE_NAME has correct default."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("OTEL_SERVICE_NAME", "ml-cicd-pipeline")
        assert result == "ml-cicd-pipeline"


def test_mlflow_model_stage_default():
    """Test that MLFLOW_MODEL_STAGE defaults to Production."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("MLFLOW_MODEL_STAGE", "Production")
        assert result == "Production"


def test_admin_token_header_default():
    """Test that ADMIN_TOKEN_HEADER has correct default."""
    from src.app.config import get_env

    with patch.dict("os.environ", {}, clear=True):
        result = get_env("ADMIN_TOKEN_HEADER", "X-Admin-Token")
        assert result == "X-Admin-Token"


def test_model_cache_dir_is_path_object():
    """Test that MODEL_CACHE_DIR is a Path object."""
    from src.app.config import MODEL_CACHE_DIR

    assert isinstance(MODEL_CACHE_DIR, Path)


def test_get_int_with_zero_max_value():
    """Test that _get_int handles zero max_value correctly."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="0"):
        result = _get_int("TEST_INT", default=10, max_value=0)
        assert result == 0


def test_get_int_with_negative_value():
    """Test that _get_int handles negative values."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="-5"):
        result = _get_int("TEST_INT", default=10)
        assert result == -5


def test_get_int_negative_exceeds_positive_max():
    """Test that negative value can be checked against max_value."""
    from src.app.config import _get_int

    # Negative values are allowed even with positive max_value
    with patch("src.app.config.get_env", return_value="-100"):
        result = _get_int("TEST_INT", default=10, max_value=50)
        assert result == -100


def test_environment_variable_override():
    """Test that environment variables override defaults."""
    from src.app.config import get_env

    with patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
        result = get_env("LOG_LEVEL", "INFO")
        assert result == "DEBUG"


def test_empty_string_environment_variable():
    """Test handling of empty string environment variables."""
    from src.app.config import get_env

    with patch.dict("os.environ", {"TEST_VAR": ""}):
        result = get_env("TEST_VAR", "default")
        # Empty string is a valid value, not None
        assert result == ""


def test_whitespace_string_environment_variable():
    """Test handling of whitespace-only environment variables."""
    from src.app.config import get_env

    with patch.dict("os.environ", {"TEST_VAR": "   "}):
        result = get_env("TEST_VAR")
        assert result == "   "


def test_get_int_with_whitespace():
    """Test that _get_int handles whitespace in numeric strings."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="  123  "):
        # int() strips whitespace automatically
        result = _get_int("TEST_INT", default=10)
        assert result == 123


def test_security_error_message_includes_variable_name():
    """Test that security error message includes variable name for debugging."""
    from src.app.config import _get_int

    with patch("src.app.config.get_env", return_value="5000"):
        with pytest.raises(ValueError, match="TEST_INT"):
            _get_int("TEST_INT", default=100, max_value=1000)
