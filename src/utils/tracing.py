"""
OpenTelemetry distributed tracing configuration.

Initializes OpenTelemetry SDK with OTLP exporter for distributed tracing.
Instruments FastAPI application automatically.
"""

from __future__ import annotations
import os
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    Resource = None
    OTLPSpanExporter = None
    FastAPIInstrumentor = None


def get_tracer(name: str) -> Any:
    """Get a tracer instance for creating spans."""
    if not OPENTELEMETRY_AVAILABLE:
        # Return a no-op tracer if OpenTelemetry is not available
        class NoOpSpan:
            """No-op span object with stub methods."""
            def set_attribute(self, *args, **kwargs):
                pass
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args, **kwargs):
                pass
        
        class NoOpTracer:
            def start_as_current_span(self, *args, **kwargs):
                return NoOpSpan()
        return NoOpTracer()
    return trace.get_tracer(name)


def initialize_tracing(
    service_name: str = "ml-cicd-pipeline",
    service_version: str = "0.1.0",
    otlp_endpoint: str | None = None,
    resource_attributes: dict[str, str] | None = None
) -> None:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for traces
        service_version: Version of the service
        otlp_endpoint: OTLP exporter endpoint (e.g., http://tempo:4317)
        resource_attributes: Additional resource attributes to include
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    # Get endpoint from environment or parameter
    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    
    # Only initialize if endpoint is provided
    if not endpoint:
        return
    
    # Build resource attributes
    attrs: dict[str, str] = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    }
    if resource_attributes:
        attrs.update(resource_attributes)
    
    # Create resource
    resource = Resource.create(attrs)
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Create OTLP exporter
    # Determine if endpoint is gRPC or HTTP based on protocol
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        # HTTP endpoint - use http exporter
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
            exporter = HTTPExporter(endpoint=endpoint)
        except ImportError:
            # Fallback to gRPC if HTTP exporter not available
            exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        # gRPC endpoint (default)
        exporter = OTLPSpanExporter(endpoint=endpoint)
    
    # Add span processor
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    """
    Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    """
    if not OPENTELEMETRY_AVAILABLE or not FastAPIInstrumentor:
        return
    
    # Only instrument if tracing is enabled
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        FastAPIInstrumentor.instrument_app(app)

