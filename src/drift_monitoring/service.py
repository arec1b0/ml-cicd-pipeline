"""
FastAPI application factory for the drift monitoring service.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest

from src.drift_monitoring.config import DriftSettings
from src.drift_monitoring.monitor import DriftMonitor


def create_app() -> FastAPI:
    """
    Build and configure the drift monitoring FastAPI application.
    """
    settings = DriftSettings.from_env()
    settings.validate()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    registry = CollectorRegistry()
    monitor = DriftMonitor(settings, registry)

    app = FastAPI(title="Drift Monitoring Service", version="0.1.0")
    app.state.monitor = monitor
    app.state.registry = registry

    @app.on_event("startup")
    async def _startup() -> None:
        await monitor.initialize()
        monitor.start_background_loop()
        logging.getLogger(__name__).info("Drift monitoring background loop started.")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await monitor.shutdown()
        logging.getLogger(__name__).info("Drift monitoring background loop stopped.")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/refresh")
    async def refresh() -> JSONResponse:
        await monitor.evaluate_once()
        return JSONResponse({"status": "triggered"})

    @app.get("/metrics")
    async def metrics() -> Response:
        payload = generate_latest(registry)
        return Response(payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
