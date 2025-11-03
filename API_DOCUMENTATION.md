# API Documentation

## Overview

This is a FastAPI-based ML inference service that provides prediction endpoints, health checks, and Prometheus metrics. The service loads a trained scikit-learn model and exposes it via REST API.

**Base URL:** `http://localhost:8000` (or as configured)

---

## Table of Contents

1. [REST API Endpoints](#rest-api-endpoints)
2. [Model Components](#model-components)
3. [Training Module](#training-module)
4. [Configuration](#configuration)
5. [Telemetry & Monitoring](#telemetry--monitoring)
6. [Quick Start Examples](#quick-start-examples)

---

## REST API Endpoints

### Health Check

**Endpoint:** `GET /health/`

**Description:** Combined liveness and readiness endpoint. Returns service status and model metrics if available.

**Response Model:**
```json
{
  "status": "ok",
  "ready": true,
  "details": {
    "metrics": {
      "accuracy": 0.95
    }
  }
}
```

**Example Request:**
```bash
curl http://localhost:8000/health/
```

**Example Response:**
```json
{
  "status": "ok",
  "ready": true,
  "details": {
    "metrics": {
      "accuracy": 0.9666666666666667
    }
  }
}
```

**Status Codes:**
- `200 OK` - Service is running (check `ready` field for model readiness)

**Fields:**
- `status` (string): Always "ok" if service is running
- `ready` (boolean): `true` if model is loaded and ready for predictions
- `details` (object, optional): Contains model metrics if available

---

### Prediction

**Endpoint:** `POST /predict/`

**Description:** Performs batch inference on feature vectors.

**Request Body:**
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5]
  ]
}
```

**Response Model:**
```json
{
  "predictions": [0, 1]
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.7, 3.1, 4.7, 1.5],
      [6.3, 2.5, 5.0, 1.9]
    ]
  }'
```

**Example Response:**
```json
{
  "predictions": [0, 1, 2]
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input format (empty features, wrong type)
- `500 Internal Server Error` - Prediction failed
- `503 Service Unavailable` - Model not loaded

**Input Validation:**
- `features` must be a list of lists
- Each feature vector must be non-empty
- All values must be numeric (floats)

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/",
    json={
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.1, 4.7, 1.5]
        ]
    }
)
predictions = response.json()["predictions"]
print(f"Predictions: {predictions}")
```

---

### Metrics

**Endpoint:** `GET /metrics/`

**Description:** Prometheus metrics endpoint. Exposes telemetry data in Prometheus text format.

**Example Request:**
```bash
curl http://localhost:8000/metrics/
```

**Example Response:**
```
# HELP ml_request_count Total HTTP requests processed
# TYPE ml_request_count counter
ml_request_count{method="POST",path="/predict/",status="200"} 42.0
# HELP ml_request_latency_seconds Request latency in seconds
# TYPE ml_request_latency_seconds histogram
ml_request_latency_seconds_bucket{le="0.005",method="POST",path="/predict/"} 38.0
# HELP ml_request_errors_total Total HTTP 5xx responses
# TYPE ml_request_errors_total counter
ml_request_errors_total{method="POST",path="/predict/"} 0.0
# HELP ml_model_accuracy Current model accuracy as reported by trainer (0.0-1.0)
# TYPE ml_model_accuracy gauge
ml_model_accuracy 0.9666666666666667
```

**Metrics Provided:**
- `ml_request_count` - Total requests (labeled by method, path, status)
- `ml_request_latency_seconds` - Request latency histogram
- `ml_request_errors_total` - Total 5xx errors
- `ml_model_accuracy` - Current model accuracy (0.0-1.0)

---

## Model Components

### ModelWrapper

**Module:** `src.models.infer`

**Description:** Wrapper around scikit-learn models providing a stable prediction API.

**Class Definition:**
```python
class ModelWrapper:
    def __init__(self, model: Any)
    def predict(self, features: Sequence[Sequence[float]]) -> list
```

**Methods:**

#### `predict(features)`

Performs batch prediction on feature vectors.

**Parameters:**
- `features` (Sequence[Sequence[float]]): 2D sequence of numeric features

**Returns:**
- `list`: Predicted labels

**Example:**
```python
from src.models.infer import load_model
from pathlib import Path

# Load model
model_wrapper = load_model(Path("model_registry/model.pkl"))

# Single prediction
prediction = model_wrapper.predict([[5.1, 3.5, 1.4, 0.2]])
print(f"Prediction: {prediction[0]}")  # Output: 0

# Batch prediction
features = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5],
    [6.3, 2.5, 5.0, 1.9]
]
predictions = model_wrapper.predict(features)
print(f"Predictions: {predictions}")  # Output: [0, 1, 2]
```

---

### load_model

**Function:** `load_model(path: Path) -> ModelWrapper`

**Description:** Loads a serialized model from disk and wraps it.

**Parameters:**
- `path` (Path): Path to the model file (.pkl)

**Returns:**
- `ModelWrapper`: Wrapped model ready for inference

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `Exception`: If model loading fails

**Example:**
```python
from src.models.infer import load_model
from pathlib import Path

try:
    model = load_model(Path("/app/model_registry/model.pkl"))
    predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])
    print(f"Loaded model, prediction: {predictions}")
except FileNotFoundError:
    print("Model file not found")
except Exception as e:
    print(f"Failed to load model: {e}")
```

---

## Training Module

### train

**Module:** `src.models.trainer`

**Function:** `train(output_path: Path) -> TrainResult`

**Description:** Trains a RandomForest classifier on the Iris dataset and persists the model and metrics.

**Parameters:**
- `output_path` (Path): Path where the trained model will be saved (.pkl)

**Returns:**
- `TrainResult`: Dataclass containing:
  - `model_path` (Path): Path where model was saved
  - `accuracy` (float): Validation accuracy (0.0-1.0)

**Side Effects:**
- Creates output directory if it doesn't exist
- Saves model to `output_path`
- Saves metrics.json to `output_path.parent / "metrics.json"`

**Example - Programmatic Usage:**
```python
from src.models.trainer import train
from pathlib import Path

# Train model
result = train(Path("models/my_model.pkl"))

print(f"Model saved to: {result.model_path}")
print(f"Accuracy: {result.accuracy:.4f}")

# Load and use the trained model
from src.models.infer import load_model
model = load_model(result.model_path)
predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])
```

**Example - Command Line:**
```bash
# Train and save model
python -m src.models.trainer --output model_registry/model.pkl

# Train with metrics output
python -m src.models.trainer \
  --output model_registry/model.pkl \
  --metrics model_registry/metrics.json
```

**TrainResult Dataclass:**
```python
@dataclass
class TrainResult:
    model_path: Path  # Where the model was saved
    accuracy: float   # Validation accuracy
```

---

## Configuration

**Module:** `src.app.config`

### Environment Variables

Configuration is loaded from environment variables with defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/model_registry/model.pkl` | Path to the serialized model file |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### get_env

**Function:** `get_env(name: str, default: str | None = None) -> str | None`

**Description:** Reads an environment variable with fallback to default.

**Parameters:**
- `name` (str): Environment variable name
- `default` (str | None): Default value if not set

**Returns:**
- `str | None`: Environment variable value or default

**Example:**
```python
from src.app.config import get_env

# Get with default
api_key = get_env("API_KEY", "default-key")

# Get without default
optional_config = get_env("OPTIONAL_SETTING")
```

### Usage Example

**Docker Compose:**
```yaml
services:
  ml-service:
    environment:
      - MODEL_PATH=/models/production_model.pkl
      - LOG_LEVEL=DEBUG
```

**Kubernetes:**
```yaml
env:
  - name: MODEL_PATH
    value: "/app/models/model_v2.pkl"
  - name: LOG_LEVEL
    value: "INFO"
```

**Local Development:**
```bash
export MODEL_PATH="./local_models/model.pkl"
export LOG_LEVEL="DEBUG"
python -m uvicorn src.app.main:app --reload
```

---

## Telemetry & Monitoring

**Module:** `src.utils.telemetry`

### PrometheusMiddleware

**Description:** FastAPI middleware that automatically collects Prometheus metrics for all requests.

**Metrics Collected:**
- Request count (by method, path, status)
- Request latency histogram
- Error count (5xx responses)

**Usage:**
```python
from fastapi import FastAPI
from src.utils.telemetry import PrometheusMiddleware

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
```

The middleware is automatically applied in `create_app()`.

---

### Prometheus Metrics

#### REQUEST_COUNT

```python
from src.utils.telemetry import REQUEST_COUNT

# Counter: ml_request_count
# Labels: method, path, status
REQUEST_COUNT.labels(method="POST", path="/predict/", status="200").inc()
```

#### REQUEST_LATENCY

```python
from src.utils.telemetry import REQUEST_LATENCY

# Histogram: ml_request_latency_seconds
# Labels: method, path
REQUEST_LATENCY.labels(method="POST", path="/predict/").observe(0.042)
```

#### REQUEST_ERRORS

```python
from src.utils.telemetry import REQUEST_ERRORS

# Counter: ml_request_errors_total
# Labels: method, path
REQUEST_ERRORS.labels(method="POST", path="/predict/").inc()
```

#### MODEL_ACCURACY

```python
from src.utils.telemetry import MODEL_ACCURACY

# Gauge: ml_model_accuracy
# Set model accuracy (0.0-1.0)
MODEL_ACCURACY.set(0.95)
```

---

### metrics_response

**Function:** `metrics_response() -> Response`

**Description:** Returns current metrics in Prometheus text format.

**Returns:**
- `Response`: Starlette Response with metrics payload

**Example:**
```python
from src.utils.telemetry import metrics_response

@app.get("/custom-metrics")
async def custom_metrics():
    return metrics_response()
```

---

## Quick Start Examples

### Running the Service

**Docker Compose:**
```bash
docker-compose up
```

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python -m src.models.trainer --output model_registry/model.pkl

# Start service
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

---

### End-to-End Python Client

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Check health
health = requests.get(f"{BASE_URL}/health/").json()
print(f"Service ready: {health['ready']}")
if health.get('details', {}).get('metrics'):
    print(f"Model accuracy: {health['details']['metrics']['accuracy']:.4f}")

# 2. Make predictions
features = [
    [5.1, 3.5, 1.4, 0.2],  # Iris setosa
    [6.7, 3.1, 4.7, 1.5],  # Iris versicolor
    [6.3, 2.5, 5.0, 1.9]   # Iris virginica
]

response = requests.post(
    f"{BASE_URL}/predict/",
    json={"features": features}
)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: Class {pred}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# 3. Check metrics
metrics = requests.get(f"{BASE_URL}/metrics/").text
print("\nMetrics sample:")
print(metrics[:500])
```

---

### Integration Testing Example

```python
from fastapi.testclient import TestClient
from src.app.main import create_app

def test_prediction_flow():
    app = create_app()
    client = TestClient(app)
    
    # Trigger startup
    with client:
        # Check health
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        
        # Make prediction
        response = client.post(
            "/predict/",
            json={"features": [[5.1, 3.5, 1.4, 0.2]]}
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 1
        assert isinstance(predictions[0], int)
        
        # Check metrics are updated
        response = client.get("/metrics/")
        assert response.status_code == 200
        assert "ml_request_count" in response.text
```

---

### Batch Processing Example

```python
import requests
import numpy as np

def batch_predict(features_batch, batch_size=100):
    """Process large datasets in batches"""
    BASE_URL = "http://localhost:8000"
    all_predictions = []
    
    for i in range(0, len(features_batch), batch_size):
        batch = features_batch[i:i + batch_size]
        response = requests.post(
            f"{BASE_URL}/predict/",
            json={"features": batch.tolist()}
        )
        if response.status_code == 200:
            all_predictions.extend(response.json()["predictions"])
        else:
            raise Exception(f"Batch {i} failed: {response.text}")
    
    return all_predictions

# Example usage
large_dataset = np.random.rand(1000, 4) * 10
predictions = batch_predict(large_dataset, batch_size=100)
print(f"Processed {len(predictions)} predictions")
```

---

### Monitoring with Prometheus

**prometheus.yml:**
```yaml
scrape_configs:
  - job_name: 'ml-inference'
    scrape_interval: 15s
    static_configs:
      - targets: ['ml-service:8000']
    metrics_path: '/metrics/'
```

**PromQL Queries:**
```promql
# Request rate (requests per second)
rate(ml_request_count[5m])

# Average latency
rate(ml_request_latency_seconds_sum[5m]) / rate(ml_request_latency_seconds_count[5m])

# Error rate
rate(ml_request_errors_total[5m])

# Current model accuracy
ml_model_accuracy
```

---

## Error Handling

### Common Error Responses

**503 Service Unavailable - Model Not Loaded:**
```json
{
  "detail": "Model not loaded"
}
```

**400 Bad Request - Invalid Input:**
```json
{
  "detail": "features[0] must be a non-empty list of numbers"
}
```

**500 Internal Server Error - Prediction Failed:**
```json
{
  "detail": "Prediction failed: <error message>"
}
```

### Retry Logic Example

```python
import requests
import time

def predict_with_retry(features, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/predict/",
                json={"features": features},
                timeout=5
            )
            response.raise_for_status()
            return response.json()["predictions"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## API Versioning

Current version: **0.1.0**

The API version is embedded in the FastAPI app metadata and can be retrieved from the OpenAPI schema:

```bash
curl http://localhost:8000/openapi.json | jq '.info.version'
```

---

## OpenAPI/Swagger Documentation

Interactive API documentation is automatically generated by FastAPI:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

These provide interactive testing capabilities and complete schema information.

---

## Support & Troubleshooting

### Model Not Loading

**Symptom:** `/health/` returns `"ready": false`

**Solutions:**
1. Check `MODEL_PATH` environment variable
2. Verify model file exists and is readable
3. Check application logs for error details
4. Ensure model was saved with compatible scikit-learn version

### High Latency

**Symptom:** Slow predictions

**Solutions:**
1. Check `/metrics/` for latency breakdown
2. Reduce batch size in requests
3. Monitor system resources (CPU, memory)
4. Consider model optimization or caching

### License

See [LICENSE](LICENSE) file for details.
