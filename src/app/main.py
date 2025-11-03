"""
FastAPI application entrypoint for inference service.

Now integrates telemetry middleware and metrics endpoint.
"""

from __future__ import annotations
from fastapi import FastAPI
import logging
from pathlib import Path
from typing import Optional
import json

from src.app.config import MODEL_PATH, LOG_LEVEL
from src.models.infer import load_model
from src.app.api.health import router as health_router
from src.app.api.predict import router as predict_router
from src.app.api.metrics import router as metrics_router
from src.utils.telemetry import PrometheusMiddleware, MODEL_ACCURACY

def create_app() -> FastAPI:
    """
    Create FastAPI app and attach routes, telemetry, and model state.
    """
    log_level = LOG_LEVEL or "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    app = FastAPI(title="ml-cicd-pipeline-inference", version="0.1.0")

    # register routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(metrics_router)

    # attach telemetry middleware
    app.add_middleware(PrometheusMiddleware)

    @app.on_event("startup")
    async def _startup():
        # initialize state with names that avoid "model_" protected namespace
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
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
                logging.getLogger().info(f"Loaded model from {model_path}, accuracy={acc}")
            except Exception as exc:
                app.state.ml_wrapper = None
                app.state.ml_metrics = None
                app.state.is_ready = False
                logging.getLogger().exception(f"Failed to load model: {exc}")
        else:
            logging.getLogger().warning(f"Model file not found at {model_path}")

    @app.on_event("shutdown")
    async def _shutdown():
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.is_ready = False
        MODEL_ACCURACY.set(0)

    return app

app = create_app()
