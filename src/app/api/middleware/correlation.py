"""
Correlation ID middleware for FastAPI.

Generates or accepts X-Correlation-ID header and stores it in request state
and logger context for distributed tracing and log correlation.
"""

from __future__ import annotations
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Callable

from src.utils.logging import correlation_id_ctx


def get_correlation_id() -> str | None:
    """Get the current request's correlation ID from context."""
    return correlation_id_ctx.get()


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that generates or accepts X-Correlation-ID header.
    
    If the header is present, it uses that value. Otherwise, generates a new UUID.
    Stores the correlation ID in request.state and sets it in context for logging.
    """

    def __init__(self, app, header_name: str = "X-Correlation-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable):
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.header_name.lower())
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store in request state for access in handlers
        request.state.correlation_id = correlation_id

        # Set in context variable for logger context
        correlation_id_ctx.set(correlation_id)

        # Add correlation ID to response headers
        response = await call_next(request)
        response.headers[self.header_name] = correlation_id

        return response

