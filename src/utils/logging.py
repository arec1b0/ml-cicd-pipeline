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

from pythonjsonlogger import jsonlogger

# Context variable for correlation ID
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationIDFilter(logging.Filter):
    """A logging filter that injects the correlation ID into log records.

    This filter retrieves the correlation ID from a context variable and
    adds it to the log record, making it available for structured logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Adds the correlation ID to the log record.

        Args:
            record: The log record to be processed.

        Returns:
            Always returns True to allow the record to be processed.
        """
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = None
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """A custom JSON log formatter.

    This formatter enriches the JSON log output with additional fields
    like `correlation_id`, `level`, `logger`, and `timestamp`.
    """

    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        """Adds custom fields to the log record.

        Args:
            log_record: The dictionary representing the log record.
            record: The original log record.
            message_dict: A dictionary containing the formatted log message.
        """
        super().add_fields(log_record, record, message_dict)
        
        # Ensure correlation_id is included
        if not log_record.get("correlation_id"):
            log_record["correlation_id"] = getattr(record, "correlation_id", None)
        
        # Add standard fields for better observability
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["timestamp"] = self.formatTime(record, self.datefmt)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configures structured logging for the application.

    This function sets up a handler that formats log messages as JSON
    and includes a filter to add a correlation ID to each log record.
    It supports both "json" and "text" log formats.

    Args:
        log_level: The desired logging level (e.g., "INFO", "DEBUG").
        log_format: The log output format, either "json" or "text".
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if log_format.lower() == "json":
        # Use JSON formatter
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        # Text format
        formatter = logging.Formatter(
            "%(asctime)s [%(correlation_id)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)

    # Add correlation ID filter
    handler.addFilter(CorrelationIDFilter())

    root_logger.addHandler(handler)

