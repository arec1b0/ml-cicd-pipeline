"""
FastAPI application entrypoint for inference service.

Responsibilities:
 - configure app
 - attach model wrapper to app.state on startup
 - register routers
 - expose uvicorn server when container runs CMD
"""

from __future__ import annotations
from fastapi import FastAPI
import logging
from pathlib import Path
from typing import Optional
import json

from src.app.config import MODEL_PATH, LOG_LEVEL
from src.models.infer import load_model  # uses joblib
from src.app.api.health import router as health_router
from src.app.api.predict import router as predict_router

def create_app() -> FastAPI:
    """
    Create and configure FastAPI app instance.
    Loads model and optional metrics at startup and exposes state flags.
    """
    log_level = LOG_LEVEL or "INFO"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    app = FastAPI(title="ml-cicd-pipeline-inference", version="0.1.0")

    # register routers
    app.include_router(health_router)
    app.include_router(predict_router)

    @app.on_event("startup")
    async def _startup():
        """
        Load model and metrics if available. Expose small set of state variables
        on app.state that are free of the reserved 'model_' prefix to avoid
        Pydantic protected namespace warnings.
        """
        # default state
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.is_ready = False

        model_path = Path(MODEL_PATH)
        if model_path.exists():
            try:
                mw = load_model(model_path)
                app.state.ml_wrapper = mw
                # try load metrics from sibling file metrics.json
                metrics_path = model_path.parent / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r", encoding="utf8") as fh:
                            app.state.ml_metrics = json.load(fh)
                    except Exception:
                        app.state.ml_metrics = None
                app.state.is_ready = True
                logging.getLogger().info(f"Loaded model from {model_path}")
            except Exception as exc:
                app.state.ml_wrapper = None
                app.state.ml_metrics = None
                app.state.is_ready = False
                logging.getLogger().exception(f"Failed to load model: {exc}")
        else:
            logging.getLogger().warning(f"Model file not found at {model_path}")

    @app.on_event("shutdown")
    async def _shutdown():
        """
        Clean shutdown hooks.
        """
        app.state.ml_wrapper = None
        app.state.ml_metrics = None
        app.state.is_ready = False

    return app

# expose app object for ASGI
app = create_app()
