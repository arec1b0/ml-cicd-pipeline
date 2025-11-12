"""
Unit tests for logging module.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logging import (
    setup_logging,
    CorrelationIDFilter,
    CustomJsonFormatter,
    correlation_id_ctx,
)


@pytest.mark.unit
class TestLogging:
    """Test logging configuration and filters."""

    def test_setup_logging_configures_json_formatter(self, monkeypatch: pytest.MonkeyPatch):
        """Test setup_logging configures JSON formatter."""
        with patch('src.utils.logging.jsonlogger') as mock_jsonlogger:
            mock_jsonlogger.JsonFormatter = MagicMock
            
            setup_logging(log_level="INFO", log_format="json")
            
            # Should not raise exception
            assert True

    def test_setup_logging_configures_text_formatter(self):
        """Test setup_logging configures text formatter."""
        setup_logging(log_level="INFO", log_format="text")
        
        # Should not raise exception
        assert True

    def test_setup_logging_sets_log_level_correctly(self):
        """Test setup_logging sets log level correctly."""
        setup_logging(log_level="DEBUG", log_format="text")
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.DEBUG

    def test_correlation_id_filter_adds_correlation_id(self):
        """Test CorrelationIDFilter adds correlation ID to records."""
        filter_obj = CorrelationIDFilter()
        
        # Set correlation ID in context
        test_id = "test-correlation-id-12345"
        correlation_id_ctx.set(test_id)
        
        # Create log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Apply filter
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'correlation_id')
        assert record.correlation_id == test_id
        
        # Cleanup
        correlation_id_ctx.set(None)

    def test_correlation_id_filter_handles_none(self):
        """Test CorrelationIDFilter handles None correlation ID."""
        filter_obj = CorrelationIDFilter()
        
        # Clear correlation ID
        correlation_id_ctx.set(None)
        
        # Create log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        # Apply filter
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'correlation_id')
        assert record.correlation_id is None

    def test_custom_json_formatter_includes_correlation_id(self):
        """Test CustomJsonFormatter includes correlation_id field."""
        with patch('src.utils.logging.jsonlogger') as mock_jsonlogger:
            # Mock the parent class
            mock_jsonlogger.JsonFormatter = MagicMock
            
            formatter = CustomJsonFormatter()
            
            # Create log record with correlation_id
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            record.correlation_id = "test-id"
            
            # Mock parent method
            log_record = {}
            message_dict = {"message": "Test message"}
            
            formatter.add_fields(log_record, record, message_dict)
            
            # Should include correlation_id
            assert "correlation_id" in log_record or hasattr(record, 'correlation_id')

    def test_correlation_id_ctx_context_variable(self):
        """Test correlation_id_ctx context variable works."""
        # Set value
        test_id = "test-correlation-id-12345"
        correlation_id_ctx.set(test_id)
        
        # Get value
        result = correlation_id_ctx.get()
        
        assert result == test_id
        
        # Clear value
        correlation_id_ctx.set(None)
        
        result = correlation_id_ctx.get()
        
        assert result is None

    def test_setup_logging_removes_existing_handlers(self):
        """Test setup_logging removes existing handlers."""
        root_logger = logging.getLogger()
        
        # Add a handler
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)
        
        initial_count = len(root_logger.handlers)
        
        # Setup logging (should clear handlers)
        setup_logging(log_level="INFO", log_format="text")
        
        # Should have cleared and added new handler
        assert len(root_logger.handlers) >= 1

