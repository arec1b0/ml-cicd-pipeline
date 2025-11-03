# API Documentation

Comprehensive documentation for all public APIs, functions, and components in the ML CI/CD Pipeline inference service.

## Table of Contents

- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Prediction](#prediction)
  - [Metrics](#metrics)
- [Application Functions](#application-functions)
- [Model Functions](#model-functions)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [Data Models](#data-models)
- [Examples](#examples)

---

## API Endpoints

### Health Check

#### `GET /health/`

Combined liveness and readiness check endpoint. Returns the service status and model readiness.

**Response Model:**
```python
{
    "status": "ok",
    "ready": bool,
    "details": {
        "metrics": {
            "accuracy": float
        }
    } | None
}
```

**Status Codes:**
- `200 OK`: Service is responding
- The `ready` field indicates if the model is loaded and ready

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
            "accuracy": 0.9667
        }
    }
}
```

**When `ready: false`:**
```json
{
    "status": "ok",
    "ready": false,
    "details": null
}
```

---

### Prediction

#### `POST /predict/`

Makes predictions using the loaded ML model. Accepts a batch of feature vectors and returns predictions.

**Request Body:**
```python
{
    "features": List[List[float]]
}
```

- `features`: 2D array where each inner array is a feature vector
- Each feature vector must be non-empty
- All values must be numeric (float)

**Response Model:**
```python
{
    "predictions": List[int]
}
```

- `predictions`: Array of integer predictions corresponding to each input feature vector

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input (empty feature vectors, wrong format)
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Prediction failed

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3],
      [4.9, 3.0, 1.4, 0.2]
    ]
  }'
```

**Example Response:**
```json
{
    "predictions": [0, 2, 0]
}
```

**Error Example (Empty Feature Vector):**
```json
{
    "detail": "features[1] must be a non-empty list of numbers"
}
```

**Error Example (Model Not Loaded):**
```json
{
    "detail": "Model not loaded"
}
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/",
    json={
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
    }
)
result = response.json()
print(result["predictions"])  # [0, 2, 0]
```

---

### Metrics

#### `GET /metrics/`

Exposes Prometheus-formatted metrics collected by the telemetry middleware.

**Response:**
- Content-Type: `text/plain; version=0.0.4; charset=utf-8`
- Prometheus exposition format

**Metrics Exposed:**

1. **`ml_request_count`** (Counter)
   - Total HTTP requests processed
   - Labels: `method`, `path`, `status`

2. **`ml_request_latency_seconds`** (Histogram)
   - Request latency in seconds
   - Labels: `method`, `path`

3. **`ml_request_errors_total`** (Counter)
   - Total HTTP 5xx responses
   - Labels: `method`, `path`

4. **`ml_model_accuracy`** (Gauge)
   - Current model accuracy (0.0-1.0)
   - Set automatically from model metrics on startup

**Example Request:**
```bash
curl http://localhost:8000/metrics/
```

**Example Response:**
```
# HELP ml_request_count Total HTTP requests processed
# TYPE ml_request_count counter
ml_request_count{method="GET",path="/health/",status="200"} 5.0
ml_request_count{method="POST",path="/predict/",status="200"} 12.0

# HELP ml_request_latency_seconds Request latency in seconds
# TYPE ml_request_latency_seconds histogram
ml_request_latency_seconds_bucket{method="GET",path="/health/",le="0.005"} 3.0
ml_request_latency_seconds_bucket{method="GET",path="/health/",le="0.01"} 5.0
ml_request_latency_seconds_sum{method="GET",path="/health/"} 0.023
ml_request_latency_seconds_count{method="GET",path="/health/"} 5.0

# HELP ml_model_accuracy Current model accuracy as reported by trainer (0.0-1.0)
# TYPE ml_model_accuracy gauge
ml_model_accuracy 0.9667
```

---

## Application Functions

### `create_app() -> FastAPI`

Creates and configures the FastAPI application instance.

**Returns:**
- `FastAPI`: Configured application with routers, middleware, and startup/shutdown handlers

**Features:**
- Registers health, predict, and metrics routers
- Attaches Prometheus telemetry middleware
- Loads model and metrics on startup
- Initializes application state

**Usage:**
```python
from src.app.main import create_app

app = create_app()

# Use with uvicorn:
# uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

**Startup Behavior:**
- Attempts to load model from `MODEL_PATH` (default: `/app/model_registry/model.pkl`)
- Loads metrics from `metrics.json` in the same directory
- Sets `app.state.is_ready = True` if model loads successfully
- Sets Prometheus `MODEL_ACCURACY` gauge if accuracy is available

**Shutdown Behavior:**
- Clears model state
- Resets accuracy gauge to 0

---

## Model Functions

### `load_model(path: Path) -> ModelWrapper`

Loads a scikit-learn model from disk and wraps it for use in the inference service.

**Parameters:**
- `path` (Path): Path to the pickled model file (`.pkl`)

**Returns:**
- `ModelWrapper`: Wrapper instance providing a stable predict API

**Raises:**
- FileNotFoundError: If the model file doesn't exist
- Exception: If the model cannot be unpickled

**Usage:**
```python
from pathlib import Path
from src.models.infer import load_model

model = load_model(Path("/path/to/model.pkl"))
predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])
```

---

### `train(output_path: Path) -> TrainResult`

Trains a RandomForest classifier on the Iris dataset and saves it to disk.

**Parameters:**
- `output_path` (Path): Path where the trained model will be saved (must be a file path, e.g., `model.pkl`)

**Returns:**
- `TrainResult`: Dataclass containing:
  - `model_path` (Path): Path where the model was saved
  - `accuracy` (float): Validation accuracy (0.0-1.0)

**Side Effects:**
- Creates parent directories if they don't exist
- Saves model as pickle file
- Saves `metrics.json` in the same directory with accuracy

**Usage:**
```python
from pathlib import Path
from src.models.trainer import train

result = train(Path("model_registry/model.pkl"))
print(f"Model saved with accuracy: {result.accuracy:.4f}")
```

**CLI Usage:**
```bash
python -m src.models.trainer \
  --output model_registry/model.pkl \
  --metrics model_registry/metrics.json
```

**Training Details:**
- Dataset: Iris (4 features, 3 classes)
- Model: RandomForestClassifier (10 estimators, random_state=42)
- Split: 80% train, 20% validation
- Random state: 42 (reproducible)

---

## Configuration

### Environment Variables

#### `MODEL_PATH`

Path to the model pickle file.

- **Default:** `/app/model_registry/model.pkl`
- **Type:** String (path)
- **Example:** `export MODEL_PATH=/workspace/models/my_model.pkl`

#### `LOG_LEVEL`

Logging level for the application.

- **Default:** `INFO`
- **Valid Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Example:** `export LOG_LEVEL=DEBUG`

### Configuration Functions

#### `get_env(name: str, default: str | None = None) -> str | None`

Reads an environment variable with an optional default value.

**Parameters:**
- `name` (str): Environment variable name
- `default` (str | None): Default value if variable is not set

**Returns:**
- `str | None`: Environment variable value or default

**Usage:**
```python
from src.app.config import get_env

model_path = get_env("MODEL_PATH", "/default/path/model.pkl")
```

---

## Utilities

### `ModelWrapper`

Wrapper class around scikit-learn models providing a stable prediction interface.

#### Constructor

```python
ModelWrapper(model: Any)
```

**Parameters:**
- `model`: Scikit-learn model instance (or any object with a `predict` method)

#### Methods

##### `predict(features: Sequence[Sequence[float]]) -> list`

Predicts labels for a batch of feature vectors.

**Parameters:**
- `features`: 2D sequence of floats (list of lists, numpy array, etc.)

**Returns:**
- `list`: List of predicted labels (integers or floats converted to list)

**Example:**
```python
from src.models.infer import ModelWrapper
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

wrapper = ModelWrapper(model)
predictions = wrapper.predict([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
# Returns: [0, 1]
```

---

### `PrometheusMiddleware`

FastAPI/Starlette middleware that automatically collects Prometheus metrics for HTTP requests.

#### Constructor

```python
PrometheusMiddleware(app)
```

**Parameters:**
- `app`: ASGI application instance

#### Metrics Collected

- **Request Count:** Incremented for every request with labels `method`, `path`, `status`
- **Request Latency:** Histogram of request durations with labels `method`, `path`
- **Request Errors:** Counter for 5xx responses with labels `method`, `path`

#### Usage

The middleware is automatically added in `create_app()`. No manual configuration needed.

```python
from src.app.main import create_app

app = create_app()
# PrometheusMiddleware is already attached
```

---

### `metrics_response() -> Response`

Returns the current Prometheus metrics in text format.

**Returns:**
- `Response`: Starlette response with Prometheus-formatted metrics

**Usage:**
```python
from src.utils.telemetry import metrics_response

response = metrics_response()
# Response with Content-Type: text/plain; version=0.0.4; charset=utf-8
```

---

### Prometheus Metrics (Global)

The following Prometheus metrics are available globally:

#### `REQUEST_COUNT`
- Type: Counter
- Labels: `method`, `path`, `status`
- Description: Total HTTP requests processed

#### `REQUEST_LATENCY`
- Type: Histogram
- Labels: `method`, `path`
- Description: Request latency in seconds

#### `REQUEST_ERRORS`
- Type: Counter
- Labels: `method`, `path`
- Description: Total HTTP 5xx responses

#### `MODEL_ACCURACY`
- Type: Gauge
- Description: Current model accuracy (0.0-1.0)
- Automatically set on startup from model metrics

---

## Data Models

### `HealthResponse`

Pydantic model for health check responses.

```python
class HealthResponse(BaseModel):
    status: str  # Always "ok"
    ready: bool  # True if model is loaded
    details: dict | None = None  # Optional metrics dict
```

### `PredictRequest`

Pydantic model for prediction requests.

```python
class PredictRequest(BaseModel):
    features: List[List[float]]  # 2D array of feature vectors
```

**Validation:**
- Each inner list must be non-empty
- All values must be numeric (float)

### `PredictResponse`

Pydantic model for prediction responses.

```python
class PredictResponse(BaseModel):
    predictions: List[int]  # Integer predictions
```

### `TrainResult`

Dataclass for training function results.

```python
@dataclass
class TrainResult:
    model_path: Path  # Where the model was saved
    accuracy: float    # Validation accuracy (0.0-1.0)
```

---

## Examples

### Complete Workflow Example

#### 1. Train a Model

```python
from pathlib import Path
from src.models.trainer import train

result = train(Path("model_registry/model.pkl"))
print(f"Model accuracy: {result.accuracy:.4f}")
```

#### 2. Start the Service

```bash
export MODEL_PATH=model_registry/model.pkl
export LOG_LEVEL=INFO
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

#### 3. Check Health

```bash
curl http://localhost:8000/health/
```

#### 4. Make Predictions

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

#### 5. Check Metrics

```bash
curl http://localhost:8000/metrics/
```

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health/")
health = response.json()
print(f"Service ready: {health['ready']}")
print(f"Model accuracy: {health['details']['metrics']['accuracy']}")

# Make prediction
response = requests.post(
    f"{BASE_URL}/predict/",
    json={
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3],
            [4.9, 3.0, 1.4, 0.2]
        ]
    }
)
predictions = response.json()["predictions"]
print(f"Predictions: {predictions}")

# Get metrics
response = requests.get(f"{BASE_URL}/metrics/")
print(response.text[:500])  # First 500 chars of metrics
```

### Error Handling Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Try prediction before model is loaded
try:
    response = requests.post(
        f"{BASE_URL}/predict/",
        json={"features": [[1.0, 2.0, 3.0, 4.0]]}
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 503:
        print("Model not loaded yet")
    elif e.response.status_code == 400:
        print(f"Invalid input: {e.response.json()['detail']}")

# Invalid input example
try:
    response = requests.post(
        f"{BASE_URL}/predict/",
        json={"features": [[]]}  # Empty feature vector
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print(f"Validation error: {e.response.json()['detail']}")
```

### Docker Deployment Example

```bash
# Build and run
docker build -t ml-inference .
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/model_registry/model.pkl \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/model_registry:/app/model_registry \
  ml-inference

# Test
curl http://localhost:8000/health/
```

### Kubernetes Deployment Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-inference-config
data:
  MODEL_PATH: "/app/model_registry/model.pkl"
  LOG_LEVEL: "INFO"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: inference
        image: ml-inference:latest
        envFrom:
        - configMapRef:
            name: ml-inference-config
        ports:
        - containerPort: 8000
```

### Monitoring Integration

```python
# Prometheus scrape config example
scrape_configs:
  - job_name: 'ml-inference'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']

# Alert rules example
groups:
  - name: ml_inference
    rules:
      - alert: ModelNotReady
        expr: ml_model_accuracy == 0
        for: 5m
        annotations:
          summary: "ML model is not loaded"
      
      - alert: HighErrorRate
        expr: rate(ml_request_errors_total[5m]) > 0.1
        annotations:
          summary: "High error rate on ML inference service"
```

---

## Additional Notes

### Model Format

- Models must be pickled scikit-learn models (`.pkl` files)
- Models are loaded using `joblib.load()`
- The model must implement a `predict()` method that accepts a numpy array

### Feature Vector Requirements

- For the Iris dataset example: 4 features per vector
- Feature order must match training data
- All features must be numeric (float)

### Performance Considerations

- Predictions are processed in batches (multiple feature vectors per request)
- Latency metrics are automatically collected
- Model is loaded once at startup and reused for all predictions

### Security Notes

- No authentication is implemented by default
- Consider adding authentication for production deployments
- Metrics endpoint exposes internal state (consider access control)

---

## Version Information

- **Application Version:** 0.1.0
- **FastAPI:** Compatible with FastAPI framework
- **Python:** 3.11+
- **Model Format:** scikit-learn pickle files (joblib)

---

*Last Updated: 2024*
