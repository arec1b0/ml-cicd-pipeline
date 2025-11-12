"""
Shared pytest fixtures for all tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.models.infer import ModelWrapper


@pytest.fixture
def mock_model_wrapper() -> MagicMock:
    """Create a mock ModelWrapper for testing."""
    wrapper = MagicMock(spec=ModelWrapper)
    wrapper.predict.return_value = [0, 1, 2]
    wrapper.get_input_dimension.return_value = 4  # Iris dataset has 4 features
    wrapper._model = MagicMock()
    wrapper._model.feature_importances_ = [0.25, 0.25, 0.25, 0.25]
    return wrapper


@pytest.fixture
def test_app(mock_model_wrapper: MagicMock) -> FastAPI:
    """Create a test FastAPI app with mocked model."""
    app = FastAPI()
    
    # Initialize app state with loaded model
    app.state.is_ready = True
    app.state.ml_wrapper = mock_model_wrapper
    app.state.expected_feature_dimension = 4
    app.state.model_metadata = {
        "source": "local",
        "model_uri": "test://model",
        "expected_feature_dimension": 4,
    }
    
    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create a test client for FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create sample data with features
        df = pd.DataFrame({
            'feature_0': [5.1, 6.7, 5.0, 6.3],
            'feature_1': [3.5, 3.0, 3.3, 3.3],
            'feature_2': [1.4, 5.2, 1.4, 6.0],
            'feature_3': [0.2, 2.3, 0.2, 2.5],
            'target': [0, 1, 0, 1],
            'prediction': [0, 1, 0, 1],
        })
        df.to_csv(f.name, index=False)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Create a mock MLflow client."""
    client = MagicMock()
    client.get_latest_versions.return_value = [
        MagicMock(version="1", stage="Production", run_id="run123")
    ]
    client.download_artifacts.return_value = "/tmp/model"
    return client


@pytest.fixture
def mock_loki_response() -> dict:
    """Create a mock Loki API response."""
    return {
        "status": "success",
        "data": {
            "resultType": "streams",
            "result": [
                {
                    "stream": {"job": "ml-predictions"},
                    "values": [
                        [
                            "1234567890000000000",
                            '{"features": [[5.1, 3.5, 1.4, 0.2]], "predictions": [0]}',
                        ],
                        [
                            "1234567891000000000",
                            '{"features": [[6.7, 3.0, 5.2, 2.3]], "predictions": [1]}',
                        ],
                    ],
                }
            ],
        },
    }


@pytest.fixture
def sample_reference_dataframe() -> pd.DataFrame:
    """Create a sample reference dataframe for drift testing."""
    return pd.DataFrame({
        'feature_0': [5.1, 6.7, 5.0, 6.3, 5.5],
        'feature_1': [3.5, 3.0, 3.3, 3.3, 3.2],
        'feature_2': [1.4, 5.2, 1.4, 6.0, 1.5],
        'feature_3': [0.2, 2.3, 0.2, 2.5, 0.3],
        'target': [0, 1, 0, 1, 0],
        'prediction': [0, 1, 0, 1, 0],
    })


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up mock environment variables."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("CORRELATION_ID_HEADER", "X-Correlation-ID")


@pytest.fixture
def clear_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear specific environment variables for testing."""
    vars_to_clear = [
        "REFERENCE_DATASET_URI",
        "CURRENT_DATASET_URI",
        "LOKI_BASE_URL",
        "LOKI_QUERY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_MODEL_NAME",
        "ADMIN_API_TOKEN",
    ]
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)

