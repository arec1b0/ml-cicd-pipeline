"""
Structured JSON logging configuration.

Configures Python logging to output JSON format with correlation ID support.
Uses python-json-logger for structured output compatible with Loki.
"""

from __future__ import annotations
import logging
import sys
from contextvars import ContextVar
from typing import Any

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    # Fallback if python-json-logger is not installed
    jsonlogger = None

# Context variable for correlation ID
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationIDFilter(logging.Filter):
    """Logging filter that adds correlation_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id from context to log record."""
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = None
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that includes correlation_id and standard fields."""

    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Ensure correlation_id is included
        if not log_record.get("correlation_id"):
            log_record["correlation_id"] = getattr(record, "correlation_id", None)
        
        # Add standard fields for better observability
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["timestamp"] = self.formatTime(record, self.datefmt)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structured JSON logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format - "json" or "text" (default: json)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if log_format.lower() == "json" and jsonlogger:
        # Use JSON formatter
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        # Fallback to text format
        formatter = logging.Formatter(
            "%(asctime)s [%(correlation_id)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    
    # Add correlation ID filter
    handler.addFilter(CorrelationIDFilter())
    
    root_logger.addHandler(handler)

