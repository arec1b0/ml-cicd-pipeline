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
from mlflow.exceptions import MlflowException

from src.models.infer import load_model, ModelWrapper
from src.resilient_mlflow import ResilientMlflowClient, RetryConfig, CircuitBreakerConfig
from src.utils.rwlock import AsyncRWLock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDescriptor:
    """Represents a unique, addressable model version.

    This class provides a normalized representation of a model, whether it is
    sourced from a local file path or from an MLflow model registry. It is
    used to track the currently loaded model and to resolve new model versions.

    Attributes:
        source: The source of the model (e.g., "local", "mlflow").
        model_uri: The URI of the model.
        version: The model version, if applicable.
        stage: The model stage, if applicable.
        run_id: The ID of the MLflow run that produced the model, if applicable.
        local_path: The local file path to the model, if applicable.
        server_version: The version of the MLflow server, if applicable.
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
        """Generates a stable cache key for the model descriptor.

        This key is used to create a directory in the cache for storing the
        downloaded model artifacts.

        Returns:
            A string representing the cache key.
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
    """Represents a model that has been loaded into memory.

    This class encapsulates a loaded model instance, its associated metadata,
    and the path to the model artifacts.

    Attributes:
        wrapper: The `ModelWrapper` instance that provides a unified `predict` interface.
        model_file: The path to the model file (e.g., `model.pkl` or `model.onnx`).
        metrics: A dictionary of metrics associated with the model.
        accuracy: The accuracy of the model, if available.
        descriptor: The `ModelDescriptor` for the loaded model.
        artifact_path: The path to the directory containing the model artifacts.
    """
    wrapper: ModelWrapper
    model_file: Path
    metrics: Optional[Dict[str, Any]]
    accuracy: Optional[float]
    descriptor: ModelDescriptor
    artifact_path: Path


class ModelManager:
    """Manages the lifecycle of machine learning models.

    This class is responsible for discovering, downloading, caching, and loading
    models at runtime. It supports loading models from local file paths and from
    an MLflow model registry. It also provides a mechanism for hot-swapping
    models without service restarts.

    Attributes:
        current: The currently loaded model, or None if no model is loaded.
        supports_auto_refresh: A boolean indicating if the model manager supports
                               automatic polling for model updates.
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
        """Initializes the ModelManager.

        Args:
            source: The source of the model (e.g., "local", "mlflow").
            model_path: The local file path to the model.
            cache_dir: The directory to cache downloaded models.
            mlflow_model_name: The name of the model in MLflow.
            mlflow_model_stage: The stage of the model in MLflow.
            mlflow_model_version: The version of the model in MLflow.
            mlflow_tracking_uri: The URI of the MLflow tracking server.
        """
        self._source = (source or "mlflow").lower()
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._mlflow_model_name = mlflow_model_name
        self._mlflow_model_stage = mlflow_model_stage or "Production"
        self._mlflow_model_version = mlflow_model_version
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._client: Optional[ResilientMlflowClient] = None
        self._lock = AsyncRWLock()  # Read-write lock for concurrent reads
        self._current: Optional[LoadedModel] = None
        self._last_server_version: Optional[str] = None

        if self._source == "mlflow":
            # Prepare cache directory. It must be writable by the container.
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current(self) -> Optional[LoadedModel]:
        """
        Return the current loaded model state (if any).
        
        Note: This property does not use locking. For thread-safe access,
        use get_current_model() instead.
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
        """Initializes the model manager and loads the initial model.

        This method should be called once at application startup to ensure that
        a model is loaded and ready to serve predictions.

        Returns:
            The loaded model, or None if no model could be loaded.
        """
        if self._current is not None:
            return self._current
        return await self.reload(force=False)

    async def reload(self, *, force: bool) -> Optional[LoadedModel]:
        """Reloads the model.

        This method resolves the latest model descriptor, and if it is
        different from the currently loaded model, it downloads and loads the
        new model.

        Args:
            force: If True, the model artifacts will be re-downloaded even if
                   the model descriptor has not changed.

        Returns:
            The newly loaded model, or None if the model was not reloaded.
        """
        async with self._lock.write():
            descriptor = await self._resolve_descriptor()
            if descriptor is None:
                logger.warning("No model descriptor resolved for source", extra={"source": self._source})
                return None

            if self._current and self._current.descriptor == descriptor and not force:
                logger.info("Model reload skipped: descriptor unchanged", extra={"model_uri": descriptor.model_uri})
                return None

            state = await self._load_descriptor(descriptor, force=force)
            self._current = state
            return state

    async def refresh_if_needed(self) -> Optional[LoadedModel]:
        """Refreshes the model if a new version is available.

        This method is designed to be called periodically by a background task.
        It polls for the latest model descriptor, and if it is different from
        the currently loaded model, it downloads and loads the new model.

        Returns:
            The newly loaded model, or None if no new model was loaded.
        """
        if self._source != "mlflow":
            return None

        async with self._lock.write():
            descriptor = await self._resolve_descriptor()
            if descriptor is None:
                return None
            if self._current and self._current.descriptor == descriptor:
                return None
            state = await self._load_descriptor(descriptor, force=False)
            self._current = state
            return state
    
    async def get_current_model(self) -> Optional[LoadedModel]:
        """Get the current model with read lock for concurrent access.
        
        This method should be used when you need to access the model for
        prediction operations, allowing multiple concurrent reads.
        
        Returns:
            The currently loaded model, or None if no model is loaded.
        """
        async with self._lock.read():
            return self._current

    async def _resolve_descriptor(self) -> Optional[ModelDescriptor]:
        """Resolves the current model descriptor.

        This method resolves the current model descriptor based on the
        configured model source. For local models, it returns a descriptor
        for the configured file path. For MLflow models, it queries the
        MLflow server for the latest version.

        Returns:
            The resolved model descriptor, or None if no descriptor could be
            resolved.
        """
        if self._source == "local":
            return ModelDescriptor(
                source="local",
                model_uri=str(self._model_path),
                local_path=self._model_path,
            )
        if self._source == "mlflow":
            return await asyncio.to_thread(self._resolve_mlflow_descriptor)
        logger.error("Unsupported MODEL_SOURCE value", extra={"source": self._source})
        return None

    def _resolve_mlflow_descriptor(self) -> ModelDescriptor:
        """Resolves the latest MLflow model descriptor.

        This method queries the MLflow server for the latest version of the
        configured model and returns a `ModelDescriptor` for it. It uses a
        resilient MLflow client with retry and circuit breaker logic.

        Returns:
            The resolved MLflow model descriptor.

        Raises:
            RuntimeError: If the MLflow model name is not configured, or if
                          no model versions are found for the configured name
                          and stage.
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
        except (MlflowException, ConnectionError, TimeoutError, OSError) as exc:
            logger.warning(
                "Failed to fetch MLflow server version",
                extra={"error": str(exc), "error_type": type(exc).__name__},
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
        """Loads a model from a descriptor.

        This method loads a model from the specified descriptor. It delegates
        to the appropriate private method based on the descriptor's source.

        Args:
            descriptor: The descriptor of the model to load.
            force: If True, the model artifacts will be re-downloaded even if
                   they are already present in the cache.

        Returns:
            The loaded model.

        Raises:
            RuntimeError: If the descriptor source is unsupported.
        """
        if descriptor.source == "local":
            return await self._load_local(descriptor)
        if descriptor.source == "mlflow":
            return await self._load_mlflow(descriptor, force=force)
        raise RuntimeError(f"Unsupported descriptor source: {descriptor.source}")

    async def _load_local(self, descriptor: ModelDescriptor) -> LoadedModel:
        """Loads a model from a local file path.

        Args:
            descriptor: The descriptor of the model to load.

        Returns:
            The loaded model.

        Raises:
            FileNotFoundError: If the model path does not exist.
        """
        path = descriptor.local_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"Local model path not found: {path}")

        wrapper = await asyncio.to_thread(load_model, path)
        try:
            metrics, accuracy = await asyncio.to_thread(self._load_metrics_from_directory, path.parent)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load metrics, continuing without metrics",
                extra={"path": str(path.parent), "error": str(exc), "error_type": type(exc).__name__}
            )
            metrics, accuracy = None, None

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
        """Loads a model from MLflow.

        This method downloads the model artifacts from MLflow, caches them
        locally, and then loads the model into memory.

        Args:
            descriptor: The descriptor of the model to load.
            force: If True, the model artifacts will be re-downloaded even if
                   they are already present in the cache.

        Returns:
            The loaded model.
        """
        target_dir = self._cache_dir / descriptor.cache_key

        if target_dir.exists() and force:
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if force or not any(target_dir.iterdir()):
            await asyncio.to_thread(self._download_artifacts, descriptor.model_uri, target_dir)

        model_file = await asyncio.to_thread(self._resolve_model_file, target_dir)
        wrapper = await asyncio.to_thread(load_model, model_file)
        try:
            metrics, accuracy = await asyncio.to_thread(self._load_metrics_from_directory, target_dir)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load metrics, continuing without metrics",
                extra={"path": str(target_dir), "error": str(exc), "error_type": type(exc).__name__}
            )
            metrics, accuracy = None, None

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
        """Locates the model file within a directory of model artifacts.

        This method searches for a model file in the specified directory. It
        prioritizes ONNX models over scikit-learn models.

        Args:
            root: The root directory of the model artifacts.

        Returns:
            The path to the model file.

        Raises:
            FileNotFoundError: If no model file is found.
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
        
        Returns:
            Tuple of (metrics dict, accuracy float). Both are None if metrics file doesn't exist.
            
        Raises:
            OSError: If metrics file exists but cannot be read.
            json.JSONDecodeError: If metrics file exists but contains invalid JSON.
        """
        metrics_path = self._find_metrics_file(root)
        if not metrics_path:
            # Metrics file doesn't exist - this is fine, return None
            return None, None
        
        # Metrics file exists - try to load it, raise on error
        try:
            with open(metrics_path, "r", encoding="utf8") as handle:
                metrics = json.load(handle)
        except (OSError, IOError) as exc:
            logger.error(
                "Failed to read metrics.json file",
                extra={"path": str(metrics_path), "error": str(exc), "error_type": type(exc).__name__}
            )
            raise
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in metrics.json file",
                extra={"path": str(metrics_path), "error": str(exc), "error_type": type(exc).__name__}
            )
            raise

        accuracy = None
        if isinstance(metrics, dict) and "accuracy" in metrics:
            try:
                accuracy = float(metrics["accuracy"])
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Failed to parse accuracy from metrics",
                    extra={"path": str(metrics_path), "error": str(exc), "error_type": type(exc).__name__}
                )
                # Accuracy parsing failure is not critical, continue with None
        return metrics, accuracy

    def _find_metrics_file(self, root: Path) -> Optional[Path]:
        """Locates the metrics file within a directory of model artifacts.

        This method searches for a `metrics.json` file in the specified
        directory and its parent.

        Args:
            root: The root directory of the model artifacts.

        Returns:
            The path to the metrics file, or None if no metrics file is found.
        """
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
            from src.app.config import (
                MLFLOW_RETRY_MAX_ATTEMPTS,
                MLFLOW_RETRY_BACKOFF_FACTOR,
                MLFLOW_CIRCUIT_BREAKER_THRESHOLD,
                MLFLOW_CIRCUIT_BREAKER_TIMEOUT,
            )
            
            retry_config = RetryConfig(
                max_attempts=MLFLOW_RETRY_MAX_ATTEMPTS,
                backoff_factor=MLFLOW_RETRY_BACKOFF_FACTOR,
            )

            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=MLFLOW_CIRCUIT_BREAKER_THRESHOLD,
                timeout=MLFLOW_CIRCUIT_BREAKER_TIMEOUT,
            )

            self._client = ResilientMlflowClient(
                tracking_uri=self._mlflow_tracking_uri,
                retry_config=retry_config,
                circuit_breaker_config=circuit_breaker_config,
            )
        return self._client
