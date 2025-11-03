"""
FastAPI application entrypoint for inference service.

Now integrates telemetry middleware and metrics endpoint.
"""

from __future__ import annotations
from fastapi import FastAPI
import logging
from pathlib import Path
import json

from src.app.config import (
    MODEL_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
    CORRELATION_ID_HEADER,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME,
    OTEL_RESOURCE_ATTRIBUTES,
)
from src.models.infer import load_model
from src.app.api.health import router as health_router
from src.app.api.predict import router as predict_router
from src.app.api.metrics import router as metrics_router
from src.utils.telemetry import PrometheusMiddleware, MODEL_ACCURACY
from src.utils.logging import setup_logging
from src.app.api.middleware.correlation import CorrelationIDMiddleware
from src.utils.tracing import initialize_tracing, instrument_fastapi

def create_app() -> FastAPI:
    """Initializes and configures the FastAPI application.

    This function sets up logging, initializes OpenTelemetry tracing,
    registers API routers, and attaches middleware for telemetry and
    correlation IDs. It also defines startup and shutdown event handlers
    to manage the model lifecycle.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    # Setup structured JSON logging
    log_level = LOG_LEVEL or "INFO"
    log_format = LOG_FORMAT or "json"
    setup_logging(log_level=log_level, log_format=log_format)
    
    # Parse resource attributes if provided
    resource_attrs = None
    if OTEL_RESOURCE_ATTRIBUTES:
        # Format: "key1=value1,key2=value2"
        resource_attrs = {}
        for pair in OTEL_RESOURCE_ATTRIBUTES.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                resource_attrs[key.strip()] = value.strip()
    
    # Initialize OpenTelemetry tracing
    if OTEL_EXPORTER_OTLP_ENDPOINT:
        initialize_tracing(
            service_name=OTEL_SERVICE_NAME or "ml-cicd-pipeline",
            service_version="0.1.0",
            otlp_endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
            resource_attributes=resource_attrs,
        )
    
    app = FastAPI(title="ml-cicd-pipeline-inference", version="0.1.0")
    
    # Instrument FastAPI with OpenTelemetry (must be before middleware registration)
    instrument_fastapi(app)

    # register routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(metrics_router)

    # attach middleware - correlation ID must be first for context
    # Note: In FastAPI/Starlette, middleware added last executes first
    # So we add PrometheusMiddleware first, then CorrelationIDMiddleware to ensure correlation ID runs first
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(CorrelationIDMiddleware, header_name=CORRELATION_ID_HEADER)

    @app.on_event("startup")
    async def _startup():
        # initialize state with names that avoid "model_" protected namespace
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.model_metadata = None
        app.state.is_ready = False

        model_path = Path(MODEL_PATH)
        if model_path.exists():
            try:
                mw = load_model(model_path)
                app.state.ml_wrapper = mw
                # load metrics if available
                metrics_path = model_path.parent / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r", encoding="utf8") as fh:
                            app.state.ml_metrics = json.load(fh)
                    except Exception:
                        app.state.ml_metrics = None
                # set model accuracy gauge if available
                acc = None
                if isinstance(app.state.ml_metrics, dict) and "accuracy" in app.state.ml_metrics:
                    try:
                        acc = float(app.state.ml_metrics["accuracy"])
                        MODEL_ACCURACY.set(acc)
                    except Exception:
                        pass
                app.state.is_ready = True
                app.state.model_metadata = {
                    "model_path": str(model_path),
                    "metrics": app.state.ml_metrics,
                }
                logger = logging.getLogger(__name__)
                logger.info(
                    "Loaded model successfully",
                    extra={
                        "model_path": str(model_path),
                        "accuracy": acc,
                        "metrics_available": app.state.ml_metrics is not None,
                    }
                )
            except Exception as exc:
                app.state.ml_wrapper = None
                app.state.ml_metrics = None
                app.state.model_metadata = None
                app.state.is_ready = False
                logger = logging.getLogger(__name__)
                logger.exception(
                    "Failed to load model",
                    extra={"model_path": str(model_path), "error": str(exc)}
                )
        else:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Model file not found",
                extra={"model_path": str(model_path)}
            )

    @app.on_event("shutdown")
    async def _shutdown():
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.model_metadata = None
        app.state.is_ready = False
        MODEL_ACCURACY.set(0)

    return app

app = create_app()
