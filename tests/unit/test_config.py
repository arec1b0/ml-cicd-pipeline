"""
Unit tests for configuration module.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.app.config import get_env, _get_int, MODEL_PATH, MODEL_SOURCE


@pytest.mark.unit
class TestConfig:
    """Test configuration module functions."""

    def test_get_env_returns_environment_variable_value(self, monkeypatch: pytest.MonkeyPatch):
        """Test get_env returns environment variable value."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        value = get_env("TEST_VAR")
        
        assert value == "test_value"

    def test_get_env_returns_default_when_not_set(self, monkeypatch: pytest.MonkeyPatch):
        """Test get_env returns default when not set."""
        monkeypatch.delenv("TEST_VAR", raising=False)
        
        value = get_env("TEST_VAR", default="default_value")
        
        assert value == "default_value"

    def test_get_env_returns_none_when_not_set_no_default(self, monkeypatch: pytest.MonkeyPatch):
        """Test get_env returns None when not set and no default."""
        monkeypatch.delenv("TEST_VAR", raising=False)
        
        value = get_env("TEST_VAR")
        
        assert value is None

    def test_get_int_parses_integer_correctly(self, monkeypatch: pytest.MonkeyPatch):
        """Test _get_int parses integer correctly."""
        monkeypatch.setenv("TEST_INT", "42")
        
        value = _get_int("TEST_INT", default=0)
        
        assert value == 42
        assert isinstance(value, int)

    def test_get_int_returns_default_when_not_set(self, monkeypatch: pytest.MonkeyPatch):
        """Test _get_int returns default when not set."""
        monkeypatch.delenv("TEST_INT", raising=False)
        
        value = _get_int("TEST_INT", default=10)
        
        assert value == 10

    def test_get_int_raises_value_error_when_exceeds_max_value(self, monkeypatch: pytest.MonkeyPatch):
        """Test _get_int raises ValueError when exceeds max_value."""
        monkeypatch.setenv("TEST_INT", "1000")
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            _get_int("TEST_INT", default=0, max_value=100)

    def test_get_int_handles_invalid_integer(self, monkeypatch: pytest.MonkeyPatch):
        """Test _get_int handles invalid integer gracefully."""
        monkeypatch.setenv("TEST_INT", "not_an_int")
        
        # Should return default when parsing fails
        value = _get_int("TEST_INT", default=10)
        
        assert value == 10

    def test_configuration_constants_load_correctly(self):
        """Test all configuration constants load correctly."""
        # These should not raise exceptions
        assert MODEL_PATH is not None
        assert MODEL_SOURCE is not None
        assert isinstance(MODEL_SOURCE, str)

    def test_get_int_with_max_value_boundary(self, monkeypatch: pytest.MonkeyPatch):
        """Test _get_int with max_value at boundary."""
        monkeypatch.setenv("TEST_INT", "100")
        
        # Should succeed at boundary
        value = _get_int("TEST_INT", default=0, max_value=100)
        
        assert value == 100
        
        # Should fail above boundary
        monkeypatch.setenv("TEST_INT", "101")
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            _get_int("TEST_INT", default=0, max_value=100)

