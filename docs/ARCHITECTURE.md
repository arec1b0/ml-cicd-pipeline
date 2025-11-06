# ML CI/CD Pipeline — Architecture

This document captures the end-to-end design of the repository: how data moves through the system, where the model is trained and packaged, how the inference service is exposed, and which automation backs the delivery workflow.

## 1. System Overview

The project is organised around a lightweight Iris classifier that demonstrates a production-style ML CI/CD pipeline.

```
raw data → validation → training → model registry → container image → Kubernetes (stable+canary) → telemetry + alerting
```

- **Training pipeline** (`src/models/trainer.py`) builds a scikit-learn `RandomForestClassifier`, persists the model with metrics, and publishes artefacts.
- **Inference service** (`src/app/main.py`) is a FastAPI application that loads the packaged model, exposes health/predict/metrics endpoints, and emits Prometheus metrics via custom middleware.
- **Automation** (GitHub Actions under `.github/workflows/`) coordinates quality gates, data validation, training, and deploy/promote steps.
- **Infrastructure** (`docker/`, `infra/helm`, `infra/monitoring`) provides containerisation, Helm manifests for Kubernetes, and Prometheus recording rules/service monitors.

## 2. Codebase Map

- `src/app/` — FastAPI service entrypoint, routing, configuration, telemetry.
- `src/models/` — Train and inference wrappers around scikit-learn models.
- `src/utils/telemetry.py` — Prometheus middleware and metrics definitions shared by the service.
- `src/drift_monitoring/` — Standalone service for data/prediction drift detection
  - `monitor.py` — Core Evidently integration and metric calculation
  - `service.py` — FastAPI application factory
  - `config.py` — Settings and validation
- `src/utils/drift.py` — Prediction logging and reference dataset utilities
- `src/utils/logging.py` — Structured JSON logging with correlation ID support
- `tests/` — Pytest suites (`tests/unit`, `tests/test_data_pipeline.py`) covering trainers, inference, and data contracts.
- `mlruns/` — Local MLflow file store (optional) used when `MLFLOW_TRACKING_URI` is not overridden.
- `scripts/` — Cross-platform helper scripts (Windows bootstrap, CI test runner, canary promotion helper).
- `ci/` — Policy documents and runbooks consumed by operational teams.
- `infra/` — Helm chart (`infra/helm/ml-model-chart/`) and monitoring manifests for cluster deployment.
- `infra/monitoring/drift-*.yaml` — Kubernetes manifests for drift service deployment
- `infra/monitoring/loki-stack-values.yaml` — Loki/Promtail Helm configuration

> **Note:** `src/data/validators` is referenced by tests and the `Data Validation` workflow. Implementors should add validators in that module or adjust tests/workflows accordingly.

## 3. Data & Model Lifecycle

1. **Exploration & feature work** happens in `notebooks/` (Jupyter notebooks for data exploration, MLflow experimentation, and evaluation).
2. **Validation** is expected in `src/data/validators` (CLI invoked by `.github/workflows/data-validation.yml`). Tests ensure processed datasets exist under `data/processed/`.
3. **Training** (`src/models/trainer.py`):
   - Pulls the Iris dataset from scikit-learn.
   - Trains a RandomForest, evaluates accuracy, and logs metrics + model artefacts to MLflow (`mlflow.sklearn.log_model`).
   - Registers the model version under `MLFLOW_MODEL_NAME`, returning a resolvable `models:/...` URI for automation.
4. **Model registry**: MLflow stores versions, stages, and metrics. GitHub Actions consume the emitted `MODEL_URI` (and MLflow webhooks emit `repository_dispatch` events) to trigger deployments.
5. **Promotion**: Deploy workflows keep the container image model-agnostic. At runtime the `ModelManager` resolves the promoted MLflow stage/version, downloads artefacts into a writable cache, validates metrics, and hot-swaps the serving model without rebuilding the image.

## 4. Inference Service Architecture

- **Entrypoint**: `src/app/main.py` constructs the FastAPI app, wires routers (`health`, `predict`, `metrics`), and registers telemetry middleware.
- **Startup lifecycle**:
  - Builds a `ModelManager` from environment variables (`MODEL_SOURCE`, `MLFLOW_MODEL_NAME`/`STAGE`, `MODEL_CACHE_DIR`, etc.).
  - Resolves the target descriptor (local path or MLflow model), downloads artefacts if necessary, loads them via `src/models/infer.load_model`, and stores wrapper/metrics on `app.state`.
  - Seeds the `ml_model_accuracy` Prometheus gauge from `metrics.json`, records MLflow connectivity metadata (server version, last verification timestamp), and optionally schedules the auto-refresh loop (`MODEL_AUTO_REFRESH_SECONDS`).
- **Request handling**:
  - `/health/`: reports readiness plus the last recorded metrics blob and MLflow connectivity diagnostics.
  - `/predict/`: validates payload shape, routes to the model wrapper, and normalises predictions.
  - `/metrics/`: serves Prometheus exposition format produced by `src/utils/telemetry.metrics_response`.
  - `/admin/reload`: secured endpoint accepting an admin token and forcing a model refresh/download for zero-downtime updates.
- **Telemetry**:
  - Middleware counts requests (`ml_request_count`), measures latency histograms, and tracks 5xx errors.
  - Accuracy gauge enables deploy-time gating and continuous monitoring.

## 5. CI/CD Automation (GitHub Actions)

- `ci-lint-test.yml`: multi-OS job running Ruff, MyPy, and pytest on pushes/PRs to `main`.
- `data-validation.yml`: on-demand or data changes; executes `python -m src.data.validators.cli --sample`.
- `model-training.yml`: trains the model against MLflow, registers a new version, and surfaces the resulting `MODEL_URI` as a workflow output.
- `deploy-canary-and-promote.yml`: parses the supplied `MODEL_URI` to configure runtime environment variables (model name/stage), builds/pushes the image, deploys a Kubernetes canary via Helm, runs smoke tests, evaluates `ml_model_accuracy ≥ 0.70`, and either promotes or rolls back.

All workflows rely on Poetry-managed dependencies (Python 3.11). Secrets configure container registry credentials and kubeconfig data.

## 6. Infrastructure Footprint

- **Container image**: `docker/Dockerfile` installs requirements from `requirements.txt`, copies application code, prepares a writable cache directory, and runs Uvicorn (`src.app.main:app`) on port 8000. Models are fetched at runtime by the `ModelManager`.
- **Drift Monitoring Service**: `docker/Dockerfile.drift-monitor` builds a separate container for drift detection. Runs FastAPI app from `src/drift_monitoring/service.py`, exposes `/healthz`, `/metrics`, and `/refresh` endpoints on port 9000 (configurable). Requires access to Loki API and reference dataset storage (S3, GCS, or local filesystem).
- **Helm chart** (`infra/helm/ml-model-chart`):
  - Deploys stable and optional canary deployments/services.
  - Ingress resources support weighted canary routing via Nginx annotations.
  - Values permit tuning replica counts, image repository/tag, canary weight, and now ship managed `Secret` objects for both `ADMIN_API_TOKEN` and MLflow credentials when the corresponding `env.*SecretValue` settings are supplied.
- **Monitoring** (`infra/monitoring`):
  - `ml-service-monitor.yaml`: Prometheus Operator `ServiceMonitor` for scraping `/metrics`.
  - `ml-recording-rules.yaml`: Recording rules for request error rate and p95 latency.
  - `test-inference-deployment.yaml`: Reference manifest bundling a minimal inference service used for monitoring tests/demos.
  - `drift-monitor-*.yaml`: Kubernetes manifests for drift monitoring service deployment and configuration.

## 7. Operational Runbooks & Scripts

- `ci/runbooks/deploy-runbook.md`: human procedure for deployment and rollback.
- `scripts/windows/deploy-canary.ps1`: Windows helper to build/push images, install Helm releases, and promote/rollback.
- `scripts/windows/setup.ps1`: Windows bootstrap (install Poetry, fallback to venv).
- `scripts/ci/run_unit_tests.sh`: Portable shell script to reproduce CI test steps locally.

## 8. Dependencies & Toolchain

- **Language/runtime**: Python 3.11.
- **Core libraries**: FastAPI, scikit-learn, pandas, prometheus-client, MLflow (for experimentation and tracking).
- **Packaging**: Poetry (`pyproject.toml`) preferred; `requirements.txt` retained for Docker builds and fallbacks.
- **Testing & linting**: Pytest, Ruff, MyPy.
- **Infrastructure tools**: Docker/Compose, Helm, kubectl, Prometheus Operator.
- **CI environment**: GitHub Actions with Linux & Windows matrices, registry authentication, and Helm-based deployments.

## 9. Drift Monitoring and Observability

The system implements comprehensive data and prediction drift monitoring by integrating structured logging, log aggregation, and statistical drift detection. This section documents the complete data flow, configuration, and troubleshooting procedures.

### 9.1 Data Flow Architecture

The drift monitoring pipeline follows this data flow:

```
Inference API → JSON Logs (stdout) → Promtail → Loki → Drift Monitor → Prometheus Metrics
                                                           ↓
                                                      Evidently Reports
```

**Components:**

- **FastAPI Inference Service** (`src/app/main.py`): Emits structured JSON logs via `src/utils/logging.py` configured with `python-json-logger`. The `CorrelationIDMiddleware` adds request tracking IDs to all log entries.

- **Prediction Logging** (`src/utils/drift.py`): The `emit_prediction_log()` function is called as a background task from the `/predict` endpoint. It logs prediction events as JSON with features, predictions, and metadata.

- **Promtail** (Kubernetes DaemonSet): Scrapes pod logs from Kubernetes using service discovery. Applies JSON parsing to extract structured fields (`correlation_id`, `level`, `logger`) and applies labels (`pod`, `namespace`, `app`, `correlation_id`, `level`, `logger`). Filters pods based on label selectors (e.g., `app=ml-inference` or `app=ml-cicd-pipeline`).

- **Loki** (Log Aggregation): Receives log streams from Promtail via HTTP push API. Stores logs with labels for efficient querying. Supports LogQL queries for filtering and extraction. Configured with retention policies (default: 7 days) and ingestion rate limits.

- **Drift Monitor Service** (`src/drift_monitoring/service.py`): Standalone FastAPI service that periodically queries Loki using LogQL or reads from file/S3 URIs. Loads reference dataset from configured storage. Executes Evidently AI reports for data drift and prediction drift detection.

- **Evidently AI** (`src/drift_monitoring/monitor.py`): Runs statistical tests comparing reference dataset to current production data. Detects feature-level drift, dataset-level drift, and prediction distribution shifts. Computes Population Stability Index (PSI) for predictions.

- **Prometheus Metrics**: Drift monitor exposes Prometheus metrics at `/metrics` endpoint. Metrics include drift status flags, drift scores, feature-level metrics, and operational metrics (row counts, last run timestamp).

### 9.2 Drift Monitoring Service Configuration

The drift monitoring service is configured via environment variables and Kubernetes ConfigMaps.

**Environment Variables:**

- `REFERENCE_DATASET_URI` (required): Location of baseline dataset. Supports:
  - Local files: `file:///data/reference/train_reference.csv` or `/data/reference/train_reference.csv`
  - S3: `s3://bucket/path/to/reference.csv`
  - GCS: `gs://bucket/path/to/reference.csv`
  - Azure Blob: `az://container/path/to/reference.csv`

- `CURRENT_DATASET_URI` (optional): Alternative to Loki for sourcing current data. Same URI formats as `REFERENCE_DATASET_URI`. If not set, drift monitor uses Loki queries.

- `LOKI_BASE_URL` (optional if `CURRENT_DATASET_URI` is set): Loki endpoint URL, e.g., `http://loki.monitoring:3100` or `http://loki.monitoring.svc.cluster.local:3100`.

- `LOKI_QUERY` (optional if `CURRENT_DATASET_URI` is set): LogQL query to filter prediction logs. Example: `{app="ml-inference"} |= "prediction"`. The query should match logs containing JSON prediction events.

- `LOKI_RANGE_MINUTES` (default: 60): Time window in minutes for collecting logs from Loki. Defines how far back to query when building current dataset.

- `DRIFT_EVALUATION_INTERVAL_SECONDS` (default: 300): How often to run drift checks in seconds. The service runs evaluations in a background loop.

- `DRIFT_MIN_ROWS` (default: 50): Minimum number of rows required in current dataset before drift evaluation runs. If fewer rows are available, evaluation is skipped.

- `DRIFT_MAX_ROWS` (optional): Maximum number of rows to analyze. Useful for performance optimization with high-volume logs. If not set, all available rows are used.

- `LOG_LEVEL` (default: INFO): Logging verbosity level. Options: DEBUG, INFO, WARNING, ERROR.

**Configuration Files:**

- `infra/monitoring/drift-monitor-config.yaml`: Kubernetes ConfigMap example showing environment variable configuration. Mounted as environment variables in the drift monitor deployment.

- `docker-compose.yml`: Local development setup with both inference API and drift monitor services. Includes volume mounts for reference datasets and environment variable examples.

**Example Kubernetes Deployment:**

The drift monitor is deployed via `infra/monitoring/drift-monitor-deployment.yaml`, which:
- Mounts the ConfigMap as environment variables
- Configures resource requests/limits
- Sets up health checks (`/healthz` endpoint)
- Exposes metrics port (9000) for Prometheus scraping
- References `infra/monitoring/drift-monitor-service.yaml` for service definition
- Uses `infra/monitoring/drift-monitor-servicemonitor.yaml` for Prometheus Operator integration

**Reference Dataset Format:**

The reference dataset CSV must contain:
- Feature columns: `feature_0`, `feature_1`, ..., `feature_N` (where N matches model input dimensions)
- Optional: `prediction` column (for prediction drift detection)
- Optional: `target` column (for target drift detection)

Example:
```csv
feature_0,feature_1,feature_2,feature_3,prediction,target
5.1,3.5,1.4,0.2,0,0
4.9,3.0,1.4,0.2,0,0
...
```

### 9.3 Log Aggregation Setup (Loki/Promtail)

**Deployment Steps:**

1. **Install Loki Stack via Helm:**

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki-stack grafana/loki-stack -n monitoring -f infra/monitoring/loki-stack-values.yaml
```

This installs Loki, Promtail, and optionally Grafana in the `monitoring` namespace.

2. **Loki Configuration Highlights:**

The `infra/monitoring/loki-stack-values.yaml` file configures:

- **Retention period**: 168h (7 days) - adjustable via `limits_config.retention_period`
- **Storage**: BoltDB shipper with filesystem backend by default. Can be configured for S3/GCS by modifying `storage_config`
- **Ingestion limits**: 10MB/s rate (`ingestion_rate_mb`), 20MB burst (`ingestion_burst_size_mb`)
- **Schema**: v11 schema with 24h index period

3. **Promtail Pipeline Stages:**

Promtail is configured with JSON parsing and label extraction:

- **JSON parsing**: Extracts `correlation_id`, `level`, `logger` from structured log messages
- **Labels applied**: `pod`, `namespace`, `app`, `correlation_id`, `level`, `logger`
- **Filtering**: Only scrapes pods with label `app=ml-cicd-pipeline` or `app=ml-model` via `relabel_configs`

Example Promtail configuration snippet:
```yaml
pipeline_stages:
  - json:
      expressions:
        correlation_id: correlation_id
        level: level
        logger: logger
  - labels:
      correlation_id:
      level:
      logger:
relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    action: keep
    regex: ml-cicd-pipeline|ml-model
```

4. **Structured Logging Setup:**

The inference API uses `src/utils/logging.py` to configure structured JSON logging:

- **Setup**: `setup_logging(log_level="INFO", log_format="json")` is called in `src/app/main.py`
- **Correlation ID**: `CorrelationIDMiddleware` extracts correlation ID from request headers and injects it into log context
- **Prediction logs**: `emit_prediction_log()` in `src/utils/drift.py` emits JSON-formatted prediction events as INFO-level logs

**Log Format Example:**

The API emits logs in this JSON format:
```json
{
  "timestamp": "2025-11-06T12:34:56",
  "level": "INFO",
  "logger": "drift-monitoring",
  "correlation_id": "abc-123",
  "message": "{\"event\": \"prediction\", \"event_id\": \"...\", \"features\": [[...]], \"predictions\": [...]}"
}
```

Note: The `message` field contains a JSON string with the prediction event payload. The drift monitor parses this nested JSON to extract features and predictions.

**Verifying Log Aggregation:**

1. Check Promtail is discovering pods:
```bash
kubectl port-forward -n monitoring svc/promtail 3101:3101
curl http://localhost:3101/targets
```

2. Query Loki for prediction logs:
```bash
curl -G 'http://loki.monitoring:3100/loki/api/v1/query_range' \
  --data-urlencode 'query={app="ml-inference"} |= "prediction"' \
  --data-urlencode 'start=...' --data-urlencode 'end=...'
```

3. Verify logs are JSON formatted:
```bash
kubectl logs -n default -l app=ml-inference | head -5 | jq .
```

### 9.4 Drift Monitoring Metrics

The drift monitor service exposes Prometheus metrics at the `/metrics` endpoint. These metrics are scraped by Prometheus via the ServiceMonitor defined in `infra/monitoring/drift-monitor-servicemonitor.yaml`.

**Prometheus Metrics Exposed:**

- `evidently_data_drift_status` (Gauge): Binary flag indicating dataset-level drift detection. Value 1 means drift detected, 0 means no drift.

- `evidently_data_drift_share` (Gauge): Percentage (0.0-1.0) of features detected as drifting. Higher values indicate more widespread drift.

- `evidently_prediction_drift_status` (Gauge): Binary flag for prediction distribution drift. Value 1 means prediction drift detected.

- `evidently_prediction_drift_score` (Gauge): Statistical test score for prediction drift. Interpretation depends on the test used by Evidently (typically Kolmogorov-Smirnov or similar).

- `evidently_prediction_psi_score` (Gauge): Population Stability Index computed for predictions. Values > 0.25 indicate significant distribution shift.

- `evidently_feature_drift_status{feature}` (Gauge): Per-feature drift detection flags. Label `feature` identifies the feature name (e.g., `feature_0`, `feature_1`). Value 1 means drift detected for that feature.

- `evidently_feature_drift_score{feature}` (Gauge): Per-feature drift scores. Higher values indicate stronger drift signals.

- `drift_monitor_current_row_count` (Gauge): Number of rows in the current production sample used for drift calculation. Zero indicates insufficient data.

- `drift_monitor_reference_row_count` (Gauge): Number of rows in the reference dataset. Set once during initialization.

- `drift_monitor_last_run_timestamp` (Gauge): Unix timestamp (UTC seconds) of the last successful drift evaluation. Useful for detecting if the monitor has stopped running.

**ServiceMonitor Setup:**

The `infra/monitoring/drift-monitor-servicemonitor.yaml` configures Prometheus Operator to scrape drift monitor metrics:

- Selects service with label `app: drift-monitor`
- Scrapes port named `metrics` (port 9000)
- Configures scrape interval and timeout
- Adds namespace label for multi-tenant setups

**Example Prometheus Queries:**

```promql
# Alert when data drift is detected
evidently_data_drift_status == 1

# Alert when drift share exceeds threshold
evidently_data_drift_share > 0.5

# Alert when prediction PSI indicates significant shift
evidently_prediction_psi_score > 0.25

# Check if monitor is running (last run > 10 minutes ago)
time() - drift_monitor_last_run_timestamp > 600

# Feature-level drift detection
evidently_feature_drift_status{feature="feature_0"} == 1
```

### 9.5 Troubleshooting Guide

This section covers common integration issues and their resolution procedures.

**Issue: Drift Monitor shows 0 current rows**

Symptoms:
- `drift_monitor_current_row_count` metric is 0
- Logs show "Not enough current data to evaluate drift"
- Drift evaluations are skipped

Diagnosis:

1. Check Loki connectivity from drift monitor pod:
```bash
kubectl exec -it <drift-pod> -- curl http://loki.monitoring:3100/ready
```

2. Test LogQL query manually:
```bash
# Get current timestamp
END=$(date +%s)000000000
START=$((END - 3600000000000))  # 1 hour ago

curl -G 'http://loki.monitoring:3100/loki/api/v1/query_range' \
  --data-urlencode "query={app=\"ml-inference\"} |= \"prediction\"" \
  --data-urlencode "start=${START}" \
  --data-urlencode "end=${END}"
```

3. Verify Promtail is scraping API pods:
```bash
kubectl logs -n monitoring -l app=promtail | grep "ml-inference"
```

4. Check API logs are JSON formatted:
```bash
kubectl logs -n default -l app=ml-inference | head -5
# Should show JSON objects, not plain text
```

Solutions:

- Ensure `LOG_FORMAT=json` environment variable is set on inference API pods
- Verify Promtail `relabel_configs` in `loki-stack-values.yaml` match pod labels (check for `app=ml-inference` or `app=ml-cicd-pipeline`)
- Check `LOKI_QUERY` syntax in drift monitor ConfigMap matches actual log format
- Confirm API pods have correct label: `app: ml-inference` (or update Promtail regex)
- Verify namespace: Promtail must be configured to discover pods in the same namespace as the API

**Issue: Drift Monitor cannot load reference dataset**

Symptoms:
- Startup fails with "Reference dataset is empty" or "Reference dataset must contain feature columns"
- Logs show "FileNotFoundError" or S3 access errors
- Service crashes on startup

Diagnosis:

1. Verify `REFERENCE_DATASET_URI` is set correctly:
```bash
kubectl get configmap drift-monitor-config -o yaml | grep REFERENCE_DATASET_URI
```

2. Check file exists and is accessible:
```bash
# For local file
kubectl exec -it <drift-pod> -- ls -la /data/reference/train_reference.csv

# For S3 (check AWS credentials)
kubectl exec -it <drift-pod> -- env | grep AWS
```

3. Validate CSV format has expected columns:
```bash
kubectl exec -it <drift-pod> -- head -5 /data/reference/train_reference.csv
# Should show: feature_0,feature_1,...,feature_N
```

Solutions:

- Reference dataset must contain columns: `feature_0`, `feature_1`, ..., `feature_N` (where N matches model input dimensions)
- Optional columns: `prediction`, `target` (for prediction/target drift detection)
- For S3: Mount AWS credentials as secrets or use IAM roles via service account annotations
- For GCS: Mount service account JSON key as secret and set `GOOGLE_APPLICATION_CREDENTIALS`
- For local files: Ensure volume mount exists and file is present in the mounted path
- Check file permissions: The drift monitor process must have read access

**Issue: High drift scores but no actual drift**

Symptoms:
- `evidently_data_drift_share` is high (>0.5)
- `evidently_prediction_psi_score` exceeds thresholds
- Manual inspection shows data looks normal

Diagnosis:

1. Check reference dataset is representative:
```bash
# Row count should be substantial (>1000 recommended)
kubectl exec -it <drift-pod> -- wc -l /data/reference/train_reference.csv
```

2. Verify feature distribution in reference vs current:
```python
# Check if reference data is outdated or biased
import pandas as pd
ref_df = pd.read_csv("train_reference.csv")
print(ref_df.describe())
# Compare with current production data statistics
```

3. Review Evidently threshold sensitivity - some statistical tests are sensitive to sample size

Solutions:

- Update reference dataset with more recent, representative data from training/validation sets
- Increase `DRIFT_MIN_ROWS` to require larger current samples (reduces false positives from small samples)
- Consider adjusting `DRIFT_MAX_ROWS` to limit outliers affecting drift calculations
- Re-train and persist new reference dataset using `src/utils/drift.persist_reference_dataset()` after model retraining
- Review Evidently documentation for threshold tuning if false positives persist

**Issue: Logs not appearing in Loki**

Symptoms:
- Loki query returns empty results
- Promtail appears healthy but no logs ingested
- Drift monitor cannot find prediction logs

Diagnosis:

1. Check Promtail scrape targets:
```bash
kubectl port-forward -n monitoring svc/promtail 3101:3101
curl http://localhost:3101/targets
# Should show API pods as active targets
```

2. Verify label selectors in `loki-stack-values.yaml`:
```yaml
relabel_configs:
  - action: keep
    regex: ml-cicd-pipeline|ml-model
    source_labels: [__meta_kubernetes_pod_label_app]
```

3. Check Loki ingestion limits:
```bash
kubectl logs -n monitoring -l app=loki | grep "rate limit"
```

4. Verify Promtail ServiceAccount has RBAC permissions:
```bash
kubectl get clusterrolebinding | grep promtail
kubectl describe clusterrole promtail
```

Solutions:

- Add correct app labels to inference pods: `app: ml-inference` (or update Promtail regex to match actual labels)
- Increase Loki `ingestion_rate_mb` in `loki-stack-values.yaml` if rate limiting occurs
- Verify namespace matches Promtail's `kubernetes_sd_configs` - Promtail must discover pods in the same namespace
- Check Promtail ServiceAccount has proper RBAC for pod discovery (needs `get`, `list`, `watch` on pods)
- Verify Promtail can access pod logs: Check node filesystem permissions and volume mounts

**Issue: Drift monitor queries but gets parsing errors**

Symptoms:
- Logs show "Skipping non-JSON log line"
- `drift_monitor_current_row_count` is lower than expected
- Some prediction events are missing from analysis

Diagnosis:

1. Examine log format from API:
```bash
kubectl logs -l app=ml-inference --tail=10
# Check if logs are pure JSON or mixed format
```

2. Check for mixed log formats (text + JSON) - third-party libraries may emit plain text logs

Solutions:

- Ensure ALL loggers use JSON format via `setup_logging(log_format="json")` in `src/app/main.py`
- Update third-party library loggers to use custom JSON handler if they bypass the root logger
- Filter out non-JSON logs in LogQL query: `{app="ml-inference"} | json` (adds JSON parsing stage)
- Review `src/utils/logging.py` to ensure all handlers use JSON formatter
- Check for log aggregation tools that might reformat logs before Promtail ingestion

**Issue: Drift monitor metrics not scraped by Prometheus**

Symptoms:
- Drift metrics don't appear in Grafana/Prometheus
- ServiceMonitor exists but no targets appear
- `/metrics` endpoint responds but Prometheus doesn't scrape

Diagnosis:

1. Check ServiceMonitor selector matches service labels:
```bash
kubectl get servicemonitor drift-monitor-sm -o yaml | grep -A5 matchLabels
kubectl get svc drift-monitor-service -o yaml | grep -A5 labels
# Labels must match
```

2. Verify Prometheus Operator is running:
```bash
kubectl get pods -n monitoring -l app=prometheus-operator
```

3. Check Prometheus logs for errors:
```bash
kubectl logs -n monitoring prometheus-... | grep drift
```

4. Verify service port name matches ServiceMonitor:
```bash
kubectl get svc drift-monitor-service -o yaml | grep -A3 ports
# Should have port named "metrics"
```

Solutions:

- Ensure service has label matching ServiceMonitor `matchLabels` (typically `app: drift-monitor`)
- Verify `port: metrics` in ServiceMonitor matches service port name (not number)
- Check ServiceMonitor namespace matches Prometheus' `serviceMonitorNamespaceSelector` - Prometheus must be configured to discover ServiceMonitors in the monitoring namespace
- Validate `/metrics` endpoint responds: `kubectl exec -it <drift-pod> -- curl http://localhost:9000/metrics`
- Check Prometheus Operator logs for ServiceMonitor reconciliation errors
- Verify Prometheus has permissions to access the service (network policies, service mesh rules)

Keep this document in sync with architectural changes. Major updates (model lifecycle, deployment strategy, monitoring stack) should be reflected here alongside runbooks and policy documentation.
