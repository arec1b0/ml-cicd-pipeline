"""
FastAPI application entrypoint for inference service.

Now integrates telemetry middleware and metrics endpoint.
"""

from __future__ import annotations
import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.app.config import (
    get_config,
    MODEL_PATH,
    MODEL_SOURCE,
    MODEL_CACHE_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    CORRELATION_ID_HEADER,
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME,
    OTEL_RESOURCE_ATTRIBUTES,
    MODEL_AUTO_REFRESH_SECONDS,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    MLFLOW_MODEL_VERSION,
    MLFLOW_TRACKING_URI,
    ADMIN_API_TOKEN,
    ADMIN_TOKEN_HEADER,
    EXPECTED_FEATURE_DIMENSION,
)
from src.app.api.health import router as health_router
from src.app.api.predict import router as predict_router
from src.app.api.explain import router as explain_router
from src.app.api.metrics import router as metrics_router
from src.app.api.admin import router as admin_router
from src.models.manager import ModelManager, LoadedModel
from src.utils.telemetry import PrometheusMiddleware, MODEL_ACCURACY
from src.utils.logging import setup_logging
from src.app.api.middleware.correlation import CorrelationIDMiddleware
from src.utils.tracing import initialize_tracing, instrument_fastapi
from src.app.api.predict import get_prediction_history
from mlflow.exceptions import MlflowException

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

    # Initialize rate limiter - 100 requests per minute per IP
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Instrument FastAPI with OpenTelemetry (must be before middleware registration)
    instrument_fastapi(app)

    # register routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(explain_router)
    app.include_router(metrics_router)
    app.include_router(admin_router)

    # attach middleware - correlation ID must be first for context
    # Note: In FastAPI/Starlette, middleware added last executes first
    # So we add PrometheusMiddleware first, then CorrelationIDMiddleware to ensure correlation ID runs first
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(CorrelationIDMiddleware, header_name=CORRELATION_ID_HEADER)

    manager = ModelManager(
        source=MODEL_SOURCE,
        model_path=MODEL_PATH,
        cache_dir=MODEL_CACHE_DIR,
        mlflow_model_name=MLFLOW_MODEL_NAME,
        mlflow_model_stage=MLFLOW_MODEL_STAGE,
        mlflow_model_version=MLFLOW_MODEL_VERSION,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )

    def _clear_state() -> None:
        """Clears the application's model-related state.

        This function resets the model wrapper, metrics, metadata, and readiness
        status. It is used during startup and shutdown to ensure a clean state.
        """
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.model_metadata = None
        app.state.is_ready = False
        app.state.expected_feature_dimension = None
        MODEL_ACCURACY.set(0)
        app.state.mlflow_connectivity = {"status": "unknown"}

    def _apply_model_state(state: LoadedModel) -> None:
        """Applies the loaded model state to the application.

        This function updates the application state with the new model wrapper,
        metrics, and metadata. It also sets the readiness flag and updates
        Prometheus metrics. It also validates and sets the expected feature dimension.

        Args:
            state: The loaded model state to apply.
        """
        log = logging.getLogger(__name__)

        # Get feature dimension from model
        model_feature_dim = state.wrapper.get_input_dimension()

        # Get the environment variable setting (if present)
        import os
        env_feature_dim = os.getenv("EXPECTED_FEATURE_DIMENSION")

        # Determine which dimension to use and validate
        if model_feature_dim is not None:
            # Use model-derived dimension
            app.state.expected_feature_dimension = model_feature_dim

            # Warn if environment variable is set but doesn't match
            if env_feature_dim is not None:
                env_dim_int = int(env_feature_dim)
                if env_dim_int != model_feature_dim:
                    log.warning(
                        "EXPECTED_FEATURE_DIMENSION mismatch: environment variable (%d) differs from model (%d). Using model dimension.",
                        env_dim_int,
                        model_feature_dim,
                        extra={
                            "env_dimension": env_dim_int,
                            "model_dimension": model_feature_dim,
                        },
                    )

            log.info(
                "Feature dimension validation complete",
                extra={
                    "expected_feature_dimension": model_feature_dim,
                    "source": "model_metadata",
                },
            )
        else:
            # Fallback to environment variable or default
            app.state.expected_feature_dimension = EXPECTED_FEATURE_DIMENSION
            log.warning(
                "Could not derive feature dimension from model, using fallback value: %d",
                EXPECTED_FEATURE_DIMENSION,
                extra={
                    "expected_feature_dimension": EXPECTED_FEATURE_DIMENSION,
                    "source": "environment_or_default",
                },
            )

        metadata: dict[str, object] = {
            "source": state.descriptor.source,
            "model_uri": state.descriptor.model_uri,
            "artifact_path": str(state.artifact_path),
            "model_file": str(state.model_file),
            "loaded_at": datetime.now(tz=timezone.utc).isoformat(),
            "expected_feature_dimension": app.state.expected_feature_dimension,
        }
        if state.descriptor.version is not None:
            metadata["version"] = state.descriptor.version
        if state.descriptor.stage is not None:
            metadata["stage"] = state.descriptor.stage
        if state.descriptor.run_id is not None:
            metadata["run_id"] = state.descriptor.run_id
        if state.accuracy is not None:
            metadata["accuracy"] = state.accuracy
        if state.descriptor.server_version is not None:
            metadata["mlflow_server_version"] = state.descriptor.server_version

        app.state.ml_wrapper = state.wrapper
        app.state.ml_metrics = state.metrics
        app.state.model_metadata = metadata
        app.state.is_ready = True
        MODEL_ACCURACY.set(state.accuracy if state.accuracy is not None else 0)
        app.state.mlflow_connectivity = {
            "status": "ok",
            "server_version": state.descriptor.server_version,
            "verified_at": datetime.now(tz=timezone.utc).isoformat(),
            "model_uri": state.descriptor.model_uri,
        }

        log.info(
            "Applied model state",
            extra={
                "model_uri": state.descriptor.model_uri,
                "version": state.descriptor.version,
                "stage": state.descriptor.stage,
                "accuracy": state.accuracy,
                "expected_feature_dimension": app.state.expected_feature_dimension,
            },
        )

    async def reload_and_apply(*, force: bool) -> Optional[LoadedModel]:
        """Reloads the model and applies the new state to the application.

        This function reloads the model from its source, and if a new model is
        loaded, it applies the new state to the application. It also handles
        errors during the reload process.

        Args:
            force: Whether to force a reload even if the model descriptor
                   has not changed.

        Returns:
            The new loaded model state, or None if the model was not reloaded.
        """
        log = logging.getLogger(__name__)
        try:
            state = await manager.reload(force=force)
        except (RuntimeError, FileNotFoundError, MlflowException, ConnectionError, OSError) as exc:
            app.state.mlflow_connectivity = {
                "status": "error",
                "error": str(exc),
                "verified_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            log.exception("Model reload failed", extra={"error": str(exc), "error_type": type(exc).__name__})
            raise
        if state:
            _apply_model_state(state)
        else:
            app.state.mlflow_connectivity = {
                "status": "ok",
                "detail": "descriptor-unchanged",
                "verified_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        return state

    async def _auto_refresh_loop(interval: int) -> None:
        """Periodically refreshes the model in the background.

        This function runs in a background task and periodically checks for a
        new version of the model. If a new version is found, it is loaded and
        applied to the application state.

        Args:
            interval: The interval in seconds between refresh attempts.
        """
        log = logging.getLogger(__name__)
        log.info("Starting model auto-refresh loop", extra={"interval_seconds": interval})
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    state = await manager.refresh_if_needed()
                except asyncio.CancelledError:
                    raise
                except (RuntimeError, FileNotFoundError, MlflowException, ConnectionError, OSError) as exc:
                    log.exception("Auto-refresh tick failed", extra={"error": str(exc), "error_type": type(exc).__name__})
                    continue
                if state:
                    _apply_model_state(state)
        except asyncio.CancelledError:
            log.info("Auto-refresh loop cancelled")
            raise

    # Initialise state attributes and shared references
    app.state.ml_wrapper = None
    app.state.ml_metrics = None
    app.state.model_metadata = None
    app.state.is_ready = False
    app.state.model_manager = manager
    app.state.admin_api_token = ADMIN_API_TOKEN
    app.state.admin_token_header = ADMIN_TOKEN_HEADER
    app.state.model_refresh_task = None
    app.state.apply_model_state = _apply_model_state
    app.state.reload_and_apply = reload_and_apply
    app.state.mlflow_connectivity = {"status": "unknown"}

    @app.on_event("startup")
    async def _startup():
        """Handles the application startup event.

        This function is called when the application starts. It validates
        configuration, clears any existing model state, performs an initial
        model load, and starts the background model refresh loop if configured.
        """
        log = logging.getLogger(__name__)
        
        # Validate configuration at startup - fail fast with clear error messages
        try:
            config = get_config()
            config.validate_config()
            log.info("Configuration validation passed")
        except ValueError as e:
            log.error("Configuration validation failed: %s", str(e))
            raise
        
        _clear_state()
        try:
            await reload_and_apply(force=True)
        except (RuntimeError, FileNotFoundError, MlflowException, ConnectionError, OSError) as exc:
            log.exception("Startup model load failed", extra={"error": str(exc), "error_type": type(exc).__name__})
            _clear_state()
            app.state.mlflow_connectivity = {
                "status": "error",
                "error": str(exc),
                "verified_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        else:
            if MODEL_AUTO_REFRESH_SECONDS > 0 and manager.supports_auto_refresh:
                log.info(
                    "Scheduling model auto-refresh",
                    extra={"interval_seconds": MODEL_AUTO_REFRESH_SECONDS},
                )
                app.state.model_refresh_task = asyncio.create_task(
                    _auto_refresh_loop(MODEL_AUTO_REFRESH_SECONDS)
                )
        
        # Initialize prediction history store if enabled
        history_store = get_prediction_history()
        if history_store:
            try:
                await history_store.initialize()
                log.info("Prediction history store initialized")
            except Exception as exc:
                log.warning(
                    "Failed to initialize prediction history store",
                    extra={"error": str(exc), "error_type": type(exc).__name__}
                )

    @app.on_event("shutdown")
    async def _shutdown():
        """Handles the application shutdown event.

        This function is called when the application shuts down. It cancels the
        background model refresh task and clears the model state.
        """
        refresh_task = getattr(app.state, "model_refresh_task", None)
        if refresh_task:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task
        
        # Close prediction history store
        history_store = get_prediction_history()
        if history_store:
            await history_store.close()
        
        _clear_state()

    return app

app = create_app()
