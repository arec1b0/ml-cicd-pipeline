"""
Tests for ModelManager blue-green deployment and atomic model swapping.

Verifies that model reloading is atomic and doesn't leave the system in a failed state.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.models.manager import ModelManager, LoadedModel, ModelDescriptor
from src.models.infer import ModelWrapper


@pytest.fixture
def tmp_model_path(tmp_path):
    """Create a temporary model file."""
    from src.models import trainer
    model_path = tmp_path / "model.pkl"
    trainer.train(model_path)
    return model_path


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.mark.asyncio
async def test_model_manager_initial_load(tmp_model_path, tmp_cache_dir):
    """Test that ModelManager loads model on first initialization."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Initially, no model is loaded
    assert manager.current is None

    # Initialize should load the model
    state = await manager.initialize()
    assert state is not None
    assert isinstance(state, LoadedModel)
    assert state.wrapper is not None
    assert manager.current is state


@pytest.mark.asyncio
async def test_model_manager_reload_with_lock(tmp_model_path, tmp_cache_dir):
    """Test that ModelManager uses lock for thread-safe reloading."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    await manager.initialize()

    # Verify lock exists and is used
    assert hasattr(manager, "_lock")
    assert isinstance(manager._lock, asyncio.Lock)

    # Test concurrent reloads (should be serialized by lock)
    tasks = [
        manager.reload(force=True),
        manager.reload(force=True),
        manager.reload(force=True),
    ]

    results = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_atomic_swap_on_successful_reload(tmp_model_path, tmp_cache_dir):
    """Test that model swap is atomic - old model stays until new one loads."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    initial_state = await manager.initialize()
    assert initial_state is not None
    initial_wrapper = initial_state.wrapper

    # Store reference to current model
    assert manager.current is initial_state
    assert manager.current.wrapper is initial_wrapper

    # Reload the model (force=True to reload even if unchanged)
    new_state = await manager.reload(force=True)
    assert new_state is not None

    # Current should now point to new state
    assert manager.current is new_state
    # New wrapper should be a different instance
    assert manager.current.wrapper is not initial_wrapper


@pytest.mark.asyncio
async def test_atomic_swap_on_failed_reload(tmp_model_path, tmp_cache_dir):
    """Test that old model stays active if reload fails."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    initial_state = await manager.initialize()
    assert initial_state is not None
    initial_wrapper = initial_state.wrapper

    # Mock _load_descriptor to fail
    with patch.object(manager, "_load_descriptor", side_effect=Exception("Load failed")):
        # Attempt reload (should fail)
        with pytest.raises(Exception, match="Load failed"):
            await manager.reload(force=True)

    # Current should still point to old state
    assert manager.current is initial_state
    assert manager.current.wrapper is initial_wrapper


@pytest.mark.asyncio
async def test_reload_skips_unchanged_descriptor(tmp_model_path, tmp_cache_dir):
    """Test that reload skips if descriptor unchanged and force=False."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    initial_state = await manager.initialize()
    assert initial_state is not None

    # Reload without force (should skip)
    new_state = await manager.reload(force=False)
    assert new_state is None  # No change

    # Current should still be initial state
    assert manager.current is initial_state


@pytest.mark.asyncio
async def test_reload_forces_refresh_when_forced(tmp_model_path, tmp_cache_dir):
    """Test that reload always reloads when force=True."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    initial_state = await manager.initialize()
    assert initial_state is not None

    # Reload with force (should always reload)
    new_state = await manager.reload(force=True)
    assert new_state is not None

    # Should be a new instance (even though same model)
    assert new_state is not initial_state


@pytest.mark.asyncio
async def test_supports_auto_refresh_for_mlflow(tmp_model_path, tmp_cache_dir):
    """Test that auto-refresh is only supported for MLflow without fixed version."""
    # Local source - no auto-refresh
    local_manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )
    assert not local_manager.supports_auto_refresh

    # MLflow with fixed version - no auto-refresh
    mlflow_fixed = ModelManager(
        source="mlflow",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name="test-model",
        mlflow_model_stage="Production",
        mlflow_model_version="1",  # Fixed version
        mlflow_tracking_uri="http://localhost:5000",
    )
    assert not mlflow_fixed.supports_auto_refresh

    # MLflow without fixed version - supports auto-refresh
    mlflow_auto = ModelManager(
        source="mlflow",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name="test-model",
        mlflow_model_stage="Production",
        mlflow_model_version=None,  # No fixed version
        mlflow_tracking_uri="http://localhost:5000",
    )
    assert mlflow_auto.supports_auto_refresh


@pytest.mark.asyncio
async def test_concurrent_reloads_are_serialized(tmp_model_path, tmp_cache_dir):
    """Test that concurrent reload calls are serialized by the lock."""
    manager = ModelManager(
        source="local",
        model_path=tmp_model_path,
        cache_dir=tmp_cache_dir,
        mlflow_model_name=None,
        mlflow_model_stage=None,
        mlflow_model_version=None,
        mlflow_tracking_uri=None,
    )

    # Load initial model
    await manager.initialize()

    # Track order of operations
    operations = []

    # Mock _load_descriptor to track calls
    original_load = manager._load_descriptor

    async def tracked_load(*args, **kwargs):
        operations.append("start_load")
        await asyncio.sleep(0.01)  # Simulate some work
        result = await original_load(*args, **kwargs)
        operations.append("end_load")
        return result

    manager._load_descriptor = tracked_load

    # Start multiple reloads concurrently
    tasks = [
        manager.reload(force=True),
        manager.reload(force=True),
    ]

    await asyncio.gather(*tasks)

    # Operations should be serialized (start, end, start, end)
    # NOT interleaved (start, start, end, end)
    assert operations == ["start_load", "end_load", "start_load", "end_load"]


@pytest.mark.asyncio
async def test_model_descriptor_comparison(tmp_model_path, tmp_cache_dir):
    """Test that model descriptor comparison works correctly."""
    descriptor1 = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/1",
        version="1",
        stage="Production",
        run_id="run-123",
    )

    descriptor2 = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/1",
        version="1",
        stage="Production",
        run_id="run-123",
    )

    descriptor3 = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/2",
        version="2",
        stage="Production",
        run_id="run-456",
    )

    # Same descriptors should be equal
    assert descriptor1 == descriptor2

    # Different descriptors should not be equal
    assert descriptor1 != descriptor3


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Test that cache keys are generated correctly for different descriptors."""
    # Local source
    local_desc = ModelDescriptor(
        source="local",
        model_uri="/path/to/model.pkl",
        local_path=Path("/path/to/model.pkl"),
    )
    assert local_desc.cache_key == "local"

    # MLflow with version
    mlflow_version = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/1",
        version="1",
        stage="Production",
    )
    assert mlflow_version.cache_key == "v1"

    # MLflow with run_id but no version
    mlflow_run = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/latest",
        run_id="abc123",
        stage="Production",
    )
    assert mlflow_run.cache_key == "run-abc123"

    # MLflow without version or run_id
    mlflow_default = ModelDescriptor(
        source="mlflow",
        model_uri="models:/test/latest",
        stage="Production",
    )
    assert mlflow_default.cache_key == "mlflow"
