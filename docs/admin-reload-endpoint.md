# Admin Model Reload Endpoint

## Overview

The `/admin/reload` endpoint enables hot model swapping without downtime. Models can be updated at runtime without restarting the inference service, supporting zero-downtime deployments and continuous model delivery.

## Endpoint Details

### URL
```
POST /admin/reload
```

### Authentication

The endpoint requires token-based authentication to prevent unauthorized model updates.

**Headers:**
```
X-Admin-Token: <your-admin-token>
```

**Configuration:**
- `ADMIN_API_TOKEN` - The secret token value (environment variable)
- `ADMIN_TOKEN_HEADER` - Custom header name (default: `X-Admin-Token`)

If `ADMIN_API_TOKEN` is not configured, the endpoint is accessible without authentication (not recommended for production).

### Request

No request body is required.

```bash
curl -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Token: your-secret-token"
```

### Response

**Success (200 OK) - Model Reloaded:**
```json
{
  "status": "reloaded",
  "detail": "Model reloaded successfully",
  "version": "2",
  "stage": "Production"
}
```

**Success (200 OK) - No Change:**
```json
{
  "status": "noop",
  "detail": "Model descriptor unchanged",
  "version": null,
  "stage": null
}
```

**Error Responses:**

- **403 Forbidden** - Invalid or missing authentication token
  ```json
  {
    "detail": "Forbidden"
  }
  ```

- **500 Internal Server Error** - Model reload failed
  ```json
  {
    "detail": "Reload failed: <error message>"
  }
  ```

- **503 Service Unavailable** - Model manager not available
  ```json
  {
    "detail": "Model manager unavailable"
  }
  ```

## Blue-Green Deployment Pattern

The endpoint implements atomic model swapping using a blue-green deployment pattern:

1. **Load New Model**: The new model is loaded and validated in memory
2. **Validate**: Model loading must succeed completely (no partial state)
3. **Atomic Swap**: Old model is replaced with new model in a single operation
4. **Cleanup**: Old model resources are released

**Key Guarantees:**
- ✅ No downtime during model updates
- ✅ Old model stays active if new model fails to load
- ✅ Atomic swap - never in a partially updated state
- ✅ Thread-safe using async locks
- ✅ Health check reflects current model state

### Implementation Details

The reload process uses:
- **AsyncIO Lock**: Ensures only one reload operation at a time
- **Blue-Green Swap**: New model loaded before old model is replaced
- **Rollback on Failure**: If loading fails, old model remains active
- **State Consistency**: App state updated atomically after successful load

## Health Check Integration

The `/health/` endpoint reflects the current model state:

```bash
curl http://localhost:8000/health/
```

**Response:**
```json
{
  "status": "ok",
  "ready": true,
  "details": {
    "metrics": {
      "accuracy": 0.95,
      "f1_score": 0.93
    }
  },
  "mlflow": {
    "status": "ok",
    "server_version": "2.7.0",
    "verified_at": "2025-11-05T12:00:00Z",
    "model_uri": "models:/iris-model/2"
  }
}
```

**Key Fields:**
- `ready`: `true` if model is loaded and ready for inference
- `details.metrics`: Model performance metrics (if available)
- `mlflow.status`: MLflow connectivity status

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ADMIN_API_TOKEN` | Secret token for authentication | None | **Recommended** |
| `ADMIN_TOKEN_HEADER` | HTTP header name for token | `X-Admin-Token` | No |
| `MODEL_SOURCE` | Model source (`local` or `mlflow`) | `mlflow` | No |
| `MODEL_PATH` | Path to local model file | `/app/model/model/model.pkl` | If `local` |
| `MODEL_CACHE_DIR` | Cache directory for MLflow models | `/var/cache/ml-model` | If `mlflow` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | None | If `mlflow` |
| `MLFLOW_MODEL_NAME` | MLflow model registry name | None | If `mlflow` |
| `MLFLOW_MODEL_STAGE` | MLflow model stage | `Production` | No |
| `MLFLOW_MODEL_VERSION` | Specific version (optional) | None | No |

### Docker Compose Example

```yaml
services:
  inference:
    image: ml-inference:latest
    environment:
      ADMIN_API_TOKEN: ${ADMIN_API_TOKEN}
      MODEL_SOURCE: mlflow
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_MODEL_NAME: iris-random-forest
      MLFLOW_MODEL_STAGE: Production
    ports:
      - "8000:8000"
```

## Usage Examples

### Basic Reload

```bash
# Reload model from MLflow
curl -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Token: secret-token-123"
```

### With Custom Header

```bash
# Using custom authentication header
export ADMIN_TOKEN_HEADER="X-Custom-Auth"

curl -X POST http://localhost:8000/admin/reload \
  -H "X-Custom-Auth: secret-token-123"
```

### Python Client

```python
import requests

def reload_model(base_url: str, admin_token: str) -> dict:
    """Reload the inference model."""
    response = requests.post(
        f"{base_url}/admin/reload",
        headers={"X-Admin-Token": admin_token}
    )
    response.raise_for_status()
    return response.json()

# Usage
result = reload_model("http://localhost:8000", "secret-token-123")
print(f"Status: {result['status']}")
print(f"Version: {result['version']}")
```

### With Health Check

```bash
#!/bin/bash
# Reload and verify health

ADMIN_TOKEN="secret-token-123"
BASE_URL="http://localhost:8000"

# Trigger reload
echo "Reloading model..."
RELOAD_RESULT=$(curl -s -X POST ${BASE_URL}/admin/reload \
  -H "X-Admin-Token: ${ADMIN_TOKEN}")

echo "Reload result: ${RELOAD_RESULT}"

# Check health
echo "Checking health..."
HEALTH_RESULT=$(curl -s ${BASE_URL}/health/)

echo "Health result: ${HEALTH_RESULT}"

# Parse ready status
READY=$(echo ${HEALTH_RESULT} | jq -r '.ready')
if [ "${READY}" = "true" ]; then
  echo "✅ Model is ready"
else
  echo "❌ Model is not ready"
  exit 1
fi
```

## Automated Reload (MLflow Auto-Refresh)

For MLflow-based deployments, you can enable automatic model refreshing:

```yaml
environment:
  MODEL_AUTO_REFRESH_SECONDS: 300  # Check every 5 minutes
  MLFLOW_MODEL_NAME: iris-model
  MLFLOW_MODEL_STAGE: Production
  # Don't set MLFLOW_MODEL_VERSION for auto-refresh
```

**How it works:**
1. Service polls MLflow every N seconds
2. Checks if a new model version is available in the specified stage
3. Automatically reloads if version changes
4. Uses same blue-green deployment pattern

**When to use:**
- ✅ Continuous delivery pipelines
- ✅ Staging environments
- ✅ A/B testing setups
- ❌ Production (use manual `/admin/reload` for control)

## Security Considerations

### Production Deployment

1. **Always set ADMIN_API_TOKEN** in production
   ```bash
   # Generate secure token
   export ADMIN_API_TOKEN=$(openssl rand -hex 32)
   ```

2. **Use HTTPS** for API communication
   ```bash
   curl -X POST https://api.example.com/admin/reload \
     -H "X-Admin-Token: ${ADMIN_API_TOKEN}"
   ```

3. **Restrict network access** to admin endpoints
   ```nginx
   # Nginx example - restrict /admin/* to internal network
   location /admin/ {
       allow 10.0.0.0/8;
       deny all;
       proxy_pass http://inference:8000;
   }
   ```

4. **Rotate tokens regularly**
   - Store tokens in secrets management (e.g., AWS Secrets Manager, Vault)
   - Rotate on a schedule (e.g., every 90 days)
   - Rotate immediately if compromised

5. **Audit logging**
   - All reload attempts are logged with correlation IDs
   - Monitor logs for unauthorized access attempts
   - Set up alerts for failed authentication

### Token Management

```python
# Example: Using AWS Secrets Manager
import boto3
import os

def get_admin_token():
    """Fetch admin token from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='ml-inference/admin-token')
    return response['SecretString']

# Set environment variable
os.environ['ADMIN_API_TOKEN'] = get_admin_token()
```

## Monitoring and Observability

### Metrics

The reload endpoint emits the following metrics:

- `ml_request_count{method="POST", path="/admin/reload", status="200"}` - Successful reloads
- `ml_request_count{method="POST", path="/admin/reload", status="403"}` - Auth failures
- `ml_request_count{method="POST", path="/admin/reload", status="500"}` - Reload errors
- `ml_request_latency_seconds` - Reload operation duration
- `ml_model_accuracy` - Current model accuracy (updated on reload)

### Logs

Structured JSON logs include:

```json
{
  "level": "info",
  "message": "Manual model reload applied",
  "model_uri": "models:/iris-model/2",
  "version": "2",
  "stage": "Production",
  "correlation_id": "abc-123-xyz",
  "timestamp": "2025-11-05T12:00:00Z"
}
```

### Grafana Dashboard Example

```promql
# Reload success rate
rate(ml_request_count{path="/admin/reload", status="200"}[5m])
/
rate(ml_request_count{path="/admin/reload"}[5m])

# Model accuracy over time
ml_model_accuracy

# Reload latency p95
histogram_quantile(0.95, ml_request_latency_seconds_bucket{path="/admin/reload"})
```

## Troubleshooting

### Issue: 403 Forbidden

**Cause:** Invalid or missing authentication token

**Solution:**
1. Verify `ADMIN_API_TOKEN` is set
2. Check header name matches `ADMIN_TOKEN_HEADER`
3. Ensure token matches exactly (no whitespace)

```bash
# Debug
echo "Token: ${ADMIN_API_TOKEN}"
curl -v -X POST http://localhost:8000/admin/reload \
  -H "X-Admin-Token: ${ADMIN_API_TOKEN}"
```

### Issue: 500 Internal Server Error

**Cause:** Model failed to load

**Solution:**
1. Check logs for detailed error message
2. Verify MLflow connectivity (`/health/` endpoint)
3. Ensure model artifact is valid
4. Check disk space in cache directory

```bash
# Check health
curl http://localhost:8000/health/ | jq '.mlflow'

# Check logs
docker logs inference-service | grep "reload"
```

### Issue: Old model still active after reload

**Cause:** New model failed to load (rollback occurred)

**Solution:**
This is expected behavior (blue-green deployment). Check logs to see why new model failed:

```bash
# Check logs for failure reason
docker logs inference-service | grep "Model reload failed"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Model Update

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true

jobs:
  reload-model:
    runs-on: ubuntu-latest
    steps:
      - name: Reload Production Model
        run: |
          curl -X POST https://api.example.com/admin/reload \
            -H "X-Admin-Token: ${{ secrets.ADMIN_API_TOKEN }}" \
            -f || exit 1

      - name: Verify Health
        run: |
          sleep 5
          READY=$(curl -s https://api.example.com/health/ | jq -r '.ready')
          if [ "$READY" != "true" ]; then
            echo "Model not ready after reload"
            exit 1
          fi

      - name: Run Smoke Tests
        run: |
          ./scripts/smoke-test.sh https://api.example.com
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any

    environment {
        ADMIN_TOKEN = credentials('ml-inference-admin-token')
        API_URL = 'https://api.example.com'
    }

    stages {
        stage('Reload Model') {
            steps {
                script {
                    def response = sh(
                        script: """
                            curl -X POST ${API_URL}/admin/reload \
                              -H "X-Admin-Token: ${ADMIN_TOKEN}" \
                              -w "%{http_code}" -o /tmp/reload.json
                        """,
                        returnStdout: true
                    ).trim()

                    if (response != '200') {
                        error("Model reload failed with status ${response}")
                    }
                }
            }
        }

        stage('Verify Health') {
            steps {
                script {
                    sleep 5
                    sh """
                        curl -s ${API_URL}/health/ | \
                        jq -e '.ready == true'
                    """
                }
            }
        }
    }
}
```

## Testing

### Unit Tests

Comprehensive unit tests are available in:
- `tests/unit/test_admin_reload.py` - Admin endpoint tests
- `tests/unit/test_model_manager.py` - ModelManager tests
- `tests/unit/test_health_with_reload.py` - Health check integration tests

### Integration Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

def test_full_reload_flow():
    """Test complete reload flow with health check."""
    client = TestClient(app)

    # Check initial health
    health = client.get("/health/")
    assert health.json()["ready"] is True
    initial_version = health.json()["mlflow"]["model_uri"]

    # Trigger reload
    reload_response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-token"}
    )
    assert reload_response.status_code == 200

    # Verify health after reload
    health = client.get("/health/")
    assert health.json()["ready"] is True

    # Test prediction still works
    predict_response = client.post(
        "/predict/",
        json={"features": [[5.1, 3.5, 1.4, 0.2]]}
    )
    assert predict_response.status_code == 200
    assert "predictions" in predict_response.json()
```

## API Reference

For complete API documentation, see the interactive API docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Related Documentation

- [Health Check API](./health-check.md)
- [Prediction API](./prediction-api.md)
- [Model Management](./model-management.md)
- [Deployment Guide](./deployment.md)
- [Security Best Practices](./security.md)
