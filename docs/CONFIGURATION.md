# Configuration Reference

This document provides a comprehensive reference for all environment variables used in the ML inference service. Configuration is managed using Pydantic BaseSettings with validation, ensuring type safety and fail-fast behavior for misconfigurations.

## Quick Start

The service uses environment variables for configuration. All variables have sensible defaults, but some are required depending on your `MODEL_SOURCE` setting.

### Required Variables by MODEL_SOURCE

#### When `MODEL_SOURCE=mlflow`:
- `MLFLOW_MODEL_NAME` - Name of the model in MLflow registry
- `MLFLOW_TRACKING_URI` - URI of the MLflow tracking server

#### When `MODEL_SOURCE=local`:
- `MODEL_PATH` - Path to the local model file (must exist)

#### Always Required:
- `ADMIN_API_TOKEN` - Admin token for administrative endpoints (required for admin API access)

## Configuration Variables

### Model Source Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `MODEL_SOURCE` | string | `mlflow` | No | Source of the model: `mlflow` or `local` |
| `MODEL_BASE_DIR` | path | `/app` | No | Base directory for model paths (used when MODEL_PATH is relative) |
| `MODEL_PATH` | path | `/app/model/model/model.pkl` | Conditional* | Path to local model file (required if MODEL_SOURCE=local) |
| `MODEL_CACHE_DIR` | path | `/var/cache/ml-model` | No | Directory to cache downloaded models |
| `MODEL_AUTO_REFRESH_SECONDS` | int | `300` | No | Interval in seconds for auto-refreshing the model (0 disables, max 3600) |
| `EXPECTED_FEATURE_DIMENSION` | int | `4` | No | Expected number of input features (default: 4 for Iris dataset, auto-derived from model at startup) |

\* `MODEL_PATH` is required when `MODEL_SOURCE=local` and must point to an existing file.

### MLflow Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `MLFLOW_MODEL_NAME` | string | `None` | Conditional* | Name of the model in MLflow registry |
| `MLFLOW_MODEL_STAGE` | string | `Production` | No | Stage of the model (e.g., `Production`, `Staging`) |
| `MLFLOW_MODEL_VERSION` | string | `None` | No | Specific model version (optional, uses stage if not set) |
| `MLFLOW_TRACKING_URI` | string | `None` | Conditional* | URI of the MLflow tracking server |
| `MLFLOW_TRACKING_USERNAME` | string | `None` | No | Username for authenticated MLflow tracking |
| `MLFLOW_TRACKING_PASSWORD` | string | `None` | No | Password for authenticated MLflow tracking |

\* Required when `MODEL_SOURCE=mlflow`

### MLflow Resilience Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `MLFLOW_RETRY_MAX_ATTEMPTS` | int | `5` | No | Maximum retry attempts for MLflow operations (1-20) |
| `MLFLOW_RETRY_BACKOFF_FACTOR` | float | `2.0` | No | Backoff multiplier for retries (1.0-10.0) |
| `MLFLOW_CIRCUIT_BREAKER_THRESHOLD` | int | `5` | No | Failure threshold for circuit breaker (1-100) |
| `MLFLOW_CIRCUIT_BREAKER_TIMEOUT` | int | `60` | No | Circuit breaker timeout in seconds (1-3600) |

### Batch Processing Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `MAX_BATCH_SIZE` | int | `1000` | No | Maximum batch size for inference requests (1-10000) |

### Logging Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `LOG_LEVEL` | string | `INFO` | No | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT` | string | `json` | No | Log format: `json` or `text` |

### Correlation ID Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `CORRELATION_ID_HEADER` | string | `X-Correlation-ID` | No | HTTP header name for correlation IDs |

### OpenTelemetry Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | string | `None` | No | OTLP endpoint for exporting OpenTelemetry traces |
| `OTEL_SERVICE_NAME` | string | `ml-cicd-pipeline` | No | Service name for OpenTelemetry tracing |
| `OTEL_RESOURCE_ATTRIBUTES` | string | `None` | No | Resource attributes in `key1=value1,key2=value2` format |

### Admin API Configuration

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `ADMIN_API_TOKEN` | string | `None` | Yes** | Admin token for accessing administrative endpoints |
| `ADMIN_TOKEN_HEADER` | string | `X-Admin-Token` | No | HTTP header name for admin token |

\** Required for admin endpoint access. Generate a secure token: `openssl rand -hex 32`

## Validation Rules

### MODEL_SOURCE Validation

The configuration system validates MODEL_SOURCE-specific requirements at startup:

- **If `MODEL_SOURCE=mlflow`**:
  - `MLFLOW_MODEL_NAME` must be set
  - `MLFLOW_TRACKING_URI` must be set

- **If `MODEL_SOURCE=local`**:
  - `MODEL_PATH` must be set
  - `MODEL_PATH` must point to an existing file

### Integer Range Validation

The following variables have range constraints:

- `MODEL_AUTO_REFRESH_SECONDS`: 0-3600 (0 disables auto-refresh)
- `MAX_BATCH_SIZE`: 1-10000
- `EXPECTED_FEATURE_DIMENSION`: ≥ 1
- `MLFLOW_RETRY_MAX_ATTEMPTS`: 1-20
- `MLFLOW_RETRY_BACKOFF_FACTOR`: 1.0-10.0
- `MLFLOW_CIRCUIT_BREAKER_THRESHOLD`: 1-100
- `MLFLOW_CIRCUIT_BREAKER_TIMEOUT`: 1-3600

### Path Validation

- `MODEL_PATH`: Must exist when `MODEL_SOURCE=local`
- `MODEL_CACHE_DIR`: Will be created if it doesn't exist (must be writable)
- `MODEL_BASE_DIR`: Used as base for relative `MODEL_PATH` values

### Format Validation

- `LOG_FORMAT`: Must be `json` or `text`
- `MODEL_SOURCE`: Must be `mlflow` or `local` (case-insensitive)

## Configuration Examples

### MLflow Deployment

```bash
export MODEL_SOURCE=mlflow
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export MLFLOW_MODEL_NAME=iris-random-forest
export MLFLOW_MODEL_STAGE=Production
export MODEL_AUTO_REFRESH_SECONDS=300
export ADMIN_API_TOKEN=$(openssl rand -hex 32)
```

### Local Model Deployment

```bash
export MODEL_SOURCE=local
export MODEL_PATH=/app/models/my-model.pkl
export ADMIN_API_TOKEN=$(openssl rand -hex 32)
```

### Custom Base Directory

```bash
export MODEL_BASE_DIR=/opt/ml-models
export MODEL_PATH=my-model.pkl  # Resolves to /opt/ml-models/my-model.pkl
```

### Development with Local MLflow

```bash
export MODEL_SOURCE=mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODEL_NAME=iris-random-forest
export MLFLOW_MODEL_STAGE=Production
export MODEL_AUTO_REFRESH_SECONDS=0  # Disable auto-refresh in dev
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
export ADMIN_API_TOKEN=dev-admin-token
```

### Production with Authentication

```bash
export MODEL_SOURCE=mlflow
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_MODEL_NAME=production-model
export MLFLOW_MODEL_STAGE=Production
export MLFLOW_TRACKING_USERNAME=mlflow-user
export MLFLOW_TRACKING_PASSWORD=secure-password
export MODEL_AUTO_REFRESH_SECONDS=300
export ADMIN_API_TOKEN=$(openssl rand -hex 32)
export LOG_LEVEL=INFO
export LOG_FORMAT=json
```

## Docker Compose Configuration

When using `docker-compose.yml`, you can set environment variables in the `environment` section:

```yaml
services:
  api:
    environment:
      - MODEL_SOURCE=mlflow
      - MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
      - MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME:-iris-random-forest}
      - MODEL_AUTO_REFRESH_SECONDS=${MODEL_AUTO_REFRESH_SECONDS:-300}
      - ADMIN_API_TOKEN=${ADMIN_API_TOKEN:?ADMIN_API_TOKEN must be set}
```

## Kubernetes/Helm Configuration

When deploying with Helm, configure values in `infra/helm/ml-model-chart/values.yaml`:

```yaml
env:
  modelSource: mlflow
  modelAutoRefreshSeconds: 300
  mlflow:
    modelName: iris-random-forest
    modelStage: Production
    trackingUri: http://mlflow:5000
  adminTokenSecretName: admin-token-secret
```

See `infra/helm/ml-model-chart/SECRET_MANAGEMENT.md` for details on secret management.

## Startup Validation

The service validates configuration at startup and will fail fast with clear error messages if:

1. Required variables are missing for the chosen `MODEL_SOURCE`
2. Integer values are outside their valid ranges
3. `MODEL_PATH` doesn't exist when `MODEL_SOURCE=local`
4. `LOG_FORMAT` is not `json` or `text`
5. `MODEL_SOURCE` is not `mlflow` or `local`

Example error message:

```
Configuration validation failed: MLFLOW_MODEL_NAME is required when MODEL_SOURCE=mlflow. 
Please set the MLFLOW_MODEL_NAME environment variable.
```

## Configuration Loading Order

1. Environment variables (highest priority)
2. `.env` file (if present in working directory)
3. Default values defined in `src/app/config.py` (lowest priority)

## Migration from Legacy Configuration

If you're upgrading from the previous configuration system:

1. All existing environment variable names remain the same
2. Default values are now centralized in `src/app/config.py`
3. `MODEL_AUTO_REFRESH_SECONDS` default changed from `0` to `300`
4. `EXPECTED_FEATURE_DIMENSION` default is now `4` (was `10` in some places)
5. Configuration validation now happens at startup (fail-fast)

## Troubleshooting

### "Configuration validation failed" Error

Check that:
- Required variables are set for your `MODEL_SOURCE`
- Integer values are within valid ranges
- File paths exist (for local models)
- Environment variable names are correct (case-insensitive)

### Model Not Loading

- Verify `MLFLOW_TRACKING_URI` is accessible
- Check `MLFLOW_MODEL_NAME` matches a registered model
- Ensure `MLFLOW_MODEL_STAGE` is correct (default: `Production`)
- For local models, verify `MODEL_PATH` exists and is readable

### Auto-Refresh Not Working

- Ensure `MODEL_AUTO_REFRESH_SECONDS > 0`
- Verify `MODEL_SOURCE=mlflow` (auto-refresh only works with MLflow)
- Check that `MLFLOW_MODEL_VERSION` is not set (version pinning disables auto-refresh)

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Overall system architecture
- [Setup Guide](SET-UP.md) - Platform-specific setup instructions
- [API Documentation](../API_DOCUMENTATION.md) - API endpoint reference
- [Helm Chart Secret Management](../infra/helm/ml-model-chart/SECRET_MANAGEMENT.md) - Kubernetes secret configuration

