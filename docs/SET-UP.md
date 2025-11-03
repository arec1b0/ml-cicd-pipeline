# Project Set-Up Guide

Use this guide to prepare a local development environment, run the inference API, and execute project checks. The instructions cover both Unix-like shells and Windows PowerShell.

## 1. Prerequisites

- Python **3.11** on your PATH (`python --version`).
- [Poetry](https://python-poetry.org/) 1.5+ for dependency management (preferred).
- Git, Docker (optional but recommended), and Make sure `pip` can reach PyPI.
- Windows: PowerShell 5+ with script execution allowed (`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).

## 2. Initial Bootstrap

### Windows 11 (recommended path)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\scripts\windows\setup.ps1
```

The script:
- Verifies Python 3.11.
- Installs Poetry if missing, then runs `poetry install`.
- Falls back to creating `.venv` and `pip install -r requirements.txt` if Poetry is unavailable.

### macOS / Linux

```bash
python -m pip install --upgrade pip
pip install poetry
poetry config virtualenvs.create false
poetry install --no-interaction --no-ansi
```

If you prefer virtual environments instead of a global site-packages install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Environment Variables

| Variable                 | Description                                                                 | Default                                   |
|--------------------------|-----------------------------------------------------------------------------|-------------------------------------------|
| `MLFLOW_TRACKING_URI`    | MLflow tracking URI (file path, HTTP(S), or Databricks URI)                 | `file://<repo-root>/mlruns` (local fallback) |
| `MLFLOW_MODEL_NAME`      | Registered model name for the trainer and deploy workflows                  | `iris-random-forest`                      |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment where training runs are recorded                          | `ml-cicd-pipeline`                        |
| `MODEL_PATH`             | Model file path used by the inference service at runtime                    | `/app/model/model/model.pkl`              |
| `LOG_LEVEL`              | Python logging level for the inference service                              | `INFO`                                    |
| `LOG_FORMAT`             | Log output format: `json` (structured) or `text` (human-readable)            | `json`                                    |
| `CORRELATION_ID_HEADER`  | HTTP header name for correlation ID (default: `X-Correlation-ID`)           | `X-Correlation-ID`                        |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP endpoint for traces (e.g., `http://tempo:4317`)      | _(not set)_                               |
| `OTEL_SERVICE_NAME`      | Service name for distributed tracing                                         | `ml-cicd-pipeline`                        |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes (format: `key1=value1,key2=value2`)    | _(not set)_                               |

When running locally via Poetry (outside Docker), point to your MLflow server and optionally override the runtime path:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODEL_NAME=iris-random-forest
export MLFLOW_EXPERIMENT_NAME=ml-cicd-pipeline
export MODEL_PATH="$(pwd)/mlruns/models--iris-random-forest/latest/model.pkl"
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=ml-cicd-pipeline
```

PowerShell equivalent:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_MODEL_NAME = "iris-random-forest"
$env:MLFLOW_EXPERIMENT_NAME = "ml-cicd-pipeline"
$env:MODEL_PATH = "$(Get-Location)\mlruns\models--iris-random-forest\latest\model.pkl"
$env:LOG_LEVEL = "INFO"
$env:LOG_FORMAT = "json"
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://localhost:4317"
$env:OTEL_SERVICE_NAME = "ml-cicd-pipeline"
```

## 4. Preparing Data & Model Artefacts

1. Ensure processed datasets exist under `data/processed/` to satisfy tests such as `tests/test_data_pipeline.py`. The repo does not ship datasets; generate them via your data preparation scripts or notebooks.
2. Export the MLflow environment variables from the previous section, then train and register a reference model:

```bash
poetry run python -m src.models.trainer --output tmp/model.pkl --metrics tmp/metrics.json
```

The `--output/--metrics` flags are optional and useful for local smoke tests; the canonical artefact lives in the MLflow Model Registry.

## 5. Running the Inference Service

### Poetry / native Python

```bash
poetry run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints:
- `GET http://127.0.0.1:8000/health/` — readiness & metrics blob.
- `POST http://127.0.0.1:8000/predict/` — batch prediction (`{"features": [[...]]}`).
- `GET http://127.0.0.1:8000/metrics/` — Prometheus exposition format.

### Docker Compose

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
MODEL_URI=models:/iris-random-forest/Production \
docker compose up --build
```

The compose file injects the MLflow credentials as build arguments, downloads the specified model during the image build, and exposes port `8000`.

### Kubernetes (Helm)

1. Package and push the image (or use CI artifact built via `deploy-canary-and-promote.yml`).
2. Set the desired registry/tag in `infra/helm/ml-model-chart/values.yaml`.
3. Install:

```bash
helm upgrade --install ml-model infra/helm/ml-model-chart \
  --set image.repository=<registry>/<repo> \
  --set image.tag=<tag>
```

Enable the canary overlay by keeping `canary.enabled=true` (default) and supplying an alternate tag (`--set canary.image.tag=<tag>`).

## 6. Quality Gates & Tooling

- **Tests**: `poetry run pytest -q`
- **Lint (Ruff)**: `poetry run ruff src tests`
- **Static typing (MyPy)**: `poetry run mypy src`
- **CI reproduction**: `./scripts/ci/run_unit_tests.sh` (POSIX) replicates the GitHub Actions verify job.

All commands assume Poetry-managed environments; drop the `poetry run` prefix if using a manually activated virtualenv.

## 7. Observability & Monitoring

### Metrics (Prometheus)

- The service exports Prometheus metrics; scrape using the provided `infra/monitoring/ml-service-monitor.yaml`.
- Recording rules for error rate and p95 latency live in `infra/monitoring/ml-recording-rules.yaml`.
- For manual Windows-based deployments with metrics gating, use `scripts/windows/deploy-canary.ps1`.

### Logs (Loki + Promtail)

The application outputs structured JSON logs with correlation IDs for distributed request tracking.

**Setup:**

1. Deploy Loki and Promtail stack:
   ```bash
   helm repo add grafana https://grafana.github.io/helm-charts
   helm repo update
   helm install loki-stack grafana/loki-stack -n monitoring \
     -f infra/monitoring/loki-stack-values.yaml \
     --create-namespace
   ```

2. Promtail will automatically collect logs from pods labeled with `app: ml-cicd-pipeline` or `app: ml-model`.

3. Logs are JSON-formatted and include:
   - `correlation_id`: Unique ID per request (from `X-Correlation-ID` header)
   - `level`: Log level (DEBUG, INFO, WARNING, ERROR)
   - `logger`: Logger name
   - `timestamp`: ISO format timestamp
   - `message`: Log message

### Traces (OpenTelemetry + Tempo/Jaeger)

The application is instrumented with OpenTelemetry for distributed tracing.

**Setup:**

1. Deploy Tempo (recommended) or Jaeger:
   ```bash
   helm install tempo grafana/tempo-distributed -n monitoring \
     -f infra/monitoring/tempo-values.yaml \
     --create-namespace
   ```

2. Configure the application to send traces:
   - Set `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (e.g., `http://tempo:4317`)
   - Update Helm values: `env.otelExporterOtlpEndpoint: "http://tempo:4317"`

3. Custom spans are created for:
   - HTTP requests (automatic via FastAPIInstrumentor)
   - Model inference (`model_inference` span with ML-specific attributes)

### Grafana Integration

**Data Sources:**

Apply the Grafana datasources configuration:
```bash
kubectl apply -f infra/monitoring/grafana-datasources.yaml
```

This configures:
- **Prometheus**: Metrics querying
- **Loki**: Log aggregation with correlation ID support
- **Tempo**: Distributed traces

**Log-to-Trace Correlation:**

Grafana is configured to correlate logs and traces:
- Click on a log entry with `correlation_id` → View related trace
- Click on a trace → View related logs for that request

**Trace Attributes:**

Model inference spans include:
- `ml.model.path`: Model file path
- `ml.input.feature_count`: Number of input features
- `ml.input.feature_dim`: Feature vector dimensions
- `ml.output.prediction_count`: Number of predictions
- `correlation.id`: Request correlation ID

## 8. Next Steps

- Review `docs/ARCHITECTURE.md` for system-wide context.
- Consult `ci/runbooks/deploy-runbook.md` before production changes.
- Keep datasets, validators, and Helm values under version control to maintain reproducibility.

## 9. MLflow Webhook Integration

To automate production deployments when a model is promoted:

1. Configure an MLflow webhook of type **Model Transitioned Stage**.
2. Point the webhook at GitHub's repository dispatch endpoint:
   ```
   POST https://api.github.com/repos/<org>/<repo>/dispatches
   ```
   with headers `Authorization: token <github-token>` and `Accept: application/vnd.github+json`.
3. Use payload:
   ```json
   {
     "event_type": "mlflow-model-promoted",
     "client_payload": {
       "model_uri": "models:/iris-random-forest/Production"
     }
   }
   ```
4. Store the GitHub token as a secret in MLflow; it must have `repo` scope to trigger the workflow.

This webhook triggers `deploy-canary-and-promote.yml`, which builds containers with the supplied `MODEL_URI` and runs the canary promotion flow.
