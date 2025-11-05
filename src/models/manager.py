"""
Model management utilities for runtime retrieval and hot-swapping.

The ModelManager encapsulates model discovery, download, validation,
and lifecycle operations so the FastAPI service can reload models
without restarting the container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow import MlflowClient

from src.models.infer import load_model, ModelWrapper
from src.resilient_mlflow import ResilientMlflowClient, RetryConfig, CircuitBreakerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDescriptor:
    """
    Normalised representation of a model target.
    """

    source: str
    model_uri: str
    version: Optional[str] = None
    stage: Optional[str] = None
    run_id: Optional[str] = None
    local_path: Optional[Path] = None
    server_version: Optional[str] = None

    @property
    def cache_key(self) -> str:
        """
        Stable cache key for storing downloaded artefacts.
        """
        if self.source == "local":
            return "local"
        if self.version:
            return f"v{self.version}"
        if self.run_id:
            return f"run-{self.run_id}"
        return "mlflow"


@dataclass
class LoadedModel:
    """
    Container for a loaded model instance and associated metadata.
    """

    wrapper: ModelWrapper
    model_file: Path
    metrics: Optional[Dict[str, Any]]
    accuracy: Optional[float]
    descriptor: ModelDescriptor
    artifact_path: Path


class ModelManager:
    """
    Handles runtime model discovery, download, caching, and swaps.
    """

    def __init__(
        self,
        *,
        source: str,
        model_path: Path,
        cache_dir: Path,
        mlflow_model_name: Optional[str],
        mlflow_model_stage: Optional[str],
        mlflow_model_version: Optional[str],
        mlflow_tracking_uri: Optional[str],
    ) -> None:
        self._source = (source or "mlflow").lower()
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._mlflow_model_name = mlflow_model_name
        self._mlflow_model_stage = mlflow_model_stage or "Production"
        self._mlflow_model_version = mlflow_model_version
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._client: Optional[ResilientMlflowClient] = None
        self._lock = asyncio.Lock()
        self._current: Optional[LoadedModel] = None
        self._last_server_version: Optional[str] = None

        if self._source == "mlflow":
            # Prepare cache directory. It must be writable by the container.
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current(self) -> Optional[LoadedModel]:
        """
        Return the current loaded model state (if any).
        """
        return self._current

    @property
    def supports_auto_refresh(self) -> bool:
        """
        Whether automatic polling for updates should be enabled.
        """
        return (
            self._source == "mlflow"
            and self._mlflow_model_name is not None
            and not self._mlflow_model_version
        )

    async def initialize(self) -> Optional[LoadedModel]:
        """
        Ensure an initial model is loaded if none present.
        """
        if self._current is not None:
            return self._current
        return await self.reload(force=False)

    async def reload(self, *, force: bool) -> Optional[LoadedModel]:
        """
        Reload the model. When force=True the artefacts are refreshed even if
        the metadata matches the currently loaded version.
        """
        async with self._lock:
            descriptor = await self._resolve_descriptor()
            if descriptor is None:
                logger.warning("No model descriptor resolved for source %s", self._source)
                return None

            if self._current and self._current.descriptor == descriptor and not force:
                logger.info("Model reload skipped: descriptor unchanged", extra={"model_uri": descriptor.model_uri})
                return None

            state = await self._load_descriptor(descriptor, force=force)
            self._current = state
            return state

    async def refresh_if_needed(self) -> Optional[LoadedModel]:
        """
        Poll for a new model and return it if a different version is available.
        """
        if self._source != "mlflow":
            return None

        async with self._lock:
            descriptor = await self._resolve_descriptor()
            if descriptor is None:
                return None
            if self._current and self._current.descriptor == descriptor:
                return None
            state = await self._load_descriptor(descriptor, force=False)
            self._current = state
            return state

    async def _resolve_descriptor(self) -> Optional[ModelDescriptor]:
        if self._source == "local":
            return ModelDescriptor(
                source="local",
                model_uri=str(self._model_path),
                local_path=self._model_path,
            )
        if self._source == "mlflow":
            return await asyncio.to_thread(self._resolve_mlflow_descriptor)
        logger.error("Unsupported MODEL_SOURCE value: %s", self._source)
        return None

    def _resolve_mlflow_descriptor(self) -> ModelDescriptor:
        """
        Resolve MLflow model descriptor using resilient client with retry logic.
        """
        if not self._mlflow_model_name:
            raise RuntimeError("MLFLOW_MODEL_NAME must be configured for mlflow source")

        client = self._get_mlflow_client()

        server_version: Optional[str] = None
        try:
            server_version = client.client.get_server_version()
            self._last_server_version = server_version
            logger.info(
                "Verified MLflow connectivity",
                extra={
                    "tracking_uri": self._mlflow_tracking_uri or mlflow.get_tracking_uri(),
                    "server_version": server_version,
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch MLflow server version",
                extra={"error": str(exc)},
            )

        if self._mlflow_model_version:
            mv = client.get_model_version(self._mlflow_model_name, self._mlflow_model_version)
        else:
            stage = self._mlflow_model_stage or "Production"
            candidates = client.get_latest_versions(self._mlflow_model_name, stages=[stage])
            if not candidates:
                raise RuntimeError(
                    f"No MLflow model versions found for name={self._mlflow_model_name} stage={stage}"
                )
            mv = candidates[0]

        model_uri = f"models:/{mv.name}/{mv.version}"
        logger.info(
            "Resolved MLflow model descriptor",
            extra={
                "model_uri": model_uri,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "server_version": server_version,
            },
        )
        return ModelDescriptor(
            source="mlflow",
            model_uri=model_uri,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            server_version=server_version,
        )

    async def _load_descriptor(self, descriptor: ModelDescriptor, *, force: bool) -> LoadedModel:
        if descriptor.source == "local":
            return await self._load_local(descriptor)
        if descriptor.source == "mlflow":
            return await self._load_mlflow(descriptor, force=force)
        raise RuntimeError(f"Unsupported descriptor source: {descriptor.source}")

    async def _load_local(self, descriptor: ModelDescriptor) -> LoadedModel:
        path = descriptor.local_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"Local model path not found: {path}")

        wrapper = await asyncio.to_thread(load_model, path)
        metrics, accuracy = await asyncio.to_thread(self._load_metrics_from_directory, path.parent)

        logger.info(
            "Loaded local model",
            extra={
                "model_path": str(path),
                "accuracy": accuracy,
            },
        )
        return LoadedModel(
            wrapper=wrapper,
            model_file=path,
            metrics=metrics,
            accuracy=accuracy,
            descriptor=descriptor,
            artifact_path=path.parent,
        )

    async def _load_mlflow(self, descriptor: ModelDescriptor, *, force: bool) -> LoadedModel:
        target_dir = self._cache_dir / descriptor.cache_key

        if target_dir.exists() and force:
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if force or not any(target_dir.iterdir()):
            await asyncio.to_thread(self._download_artifacts, descriptor.model_uri, target_dir)

        model_file = await asyncio.to_thread(self._resolve_model_file, target_dir)
        wrapper = await asyncio.to_thread(load_model, model_file)
        metrics, accuracy = await asyncio.to_thread(self._load_metrics_from_directory, target_dir)

        logger.info(
            "Loaded MLflow model",
            extra={
                "model_uri": descriptor.model_uri,
                "cache_path": str(target_dir),
                "model_file": str(model_file),
                "accuracy": accuracy,
            },
        )

        return LoadedModel(
            wrapper=wrapper,
            model_file=model_file,
            metrics=metrics,
            accuracy=accuracy,
            descriptor=descriptor,
            artifact_path=target_dir,
        )

    def _download_artifacts(self, model_uri: str, dst_path: Path) -> None:
        """
        Blocking helper to download model artefacts from MLflow with retry logic.
        """
        client = self._get_mlflow_client()

        logger.info(
            "Downloading MLflow artefacts",
            extra={
                "model_uri": model_uri,
                "dst_path": str(dst_path),
            },
        )
        client.download_artifacts(artifact_uri=model_uri, dst_path=str(dst_path))

    def _resolve_model_file(self, root: Path) -> Path:
        """
        Attempt to locate a usable model artefact inside the download directory.
        Preference order: ONNX -> sklearn pickle.
        """
        for candidate in root.rglob("model.onnx"):
            if candidate.is_file():
                return candidate
        for candidate in root.rglob("model.pkl"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No model artefact found beneath {root}")

    def _load_metrics_from_directory(self, root: Path) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        Load metrics.json if present and attempt to extract accuracy.
        """
        metrics_path = self._find_metrics_file(root)
        if not metrics_path:
            return None, None
        try:
            with open(metrics_path, "r", encoding="utf8") as handle:
                metrics = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.warning("Failed to load metrics.json", extra={"path": str(metrics_path), "error": str(exc)})
            return None, None

        accuracy = None
        if isinstance(metrics, dict) and "accuracy" in metrics:
            try:
                accuracy = float(metrics["accuracy"])
            except (TypeError, ValueError):
                accuracy = None
        return metrics, accuracy

    def _find_metrics_file(self, root: Path) -> Optional[Path]:
        candidates = [
            root / "metrics.json",
            root.parent / "metrics.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        for candidate in root.rglob("metrics.json"):
            if candidate.is_file():
                return candidate
        return None

    def _get_mlflow_client(self) -> ResilientMlflowClient:
        """Get or create resilient MLflow client with retry and circuit breaker."""
        if self._client is None:
            import os
            retry_config = RetryConfig(
                max_attempts=int(os.environ.get("MLFLOW_RETRY_MAX_ATTEMPTS", "5")),
                backoff_factor=float(os.environ.get("MLFLOW_RETRY_BACKOFF_FACTOR", "2.0")),
            )

            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=int(os.environ.get("MLFLOW_CIRCUIT_BREAKER_THRESHOLD", "5")),
                timeout=int(os.environ.get("MLFLOW_CIRCUIT_BREAKER_TIMEOUT", "60")),
            )

            self._client = ResilientMlflowClient(
                tracking_uri=self._mlflow_tracking_uri,
                retry_config=retry_config,
                circuit_breaker_config=circuit_breaker_config,
            )
        return self._client
