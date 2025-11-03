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
- `tests/` — Pytest suites (`tests/unit`, `tests/test_data_pipeline.py`) covering trainers, inference, and data contracts.
- `mlruns/` — Local MLflow file store (optional) used when `MLFLOW_TRACKING_URI` is not overridden.
- `scripts/` — Cross-platform helper scripts (Windows bootstrap, CI test runner, canary promotion helper).
- `ci/` — Policy documents and runbooks consumed by operational teams.
- `infra/` — Helm chart (`infra/helm/ml-model-chart/`) and monitoring manifests for cluster deployment.

> **Note:** `src/data/validators` is referenced by tests and the `Data Validation` workflow. Implementors should add validators in that module or adjust tests/workflows accordingly.

## 3. Data & Model Lifecycle

1. **Exploration & feature work** happens in `notebooks/` (Jupyter notebooks for data exploration, MLflow experimentation, and evaluation).
2. **Validation** is expected in `src/data/validators` (CLI invoked by `.github/workflows/data-validation.yml`). Tests ensure processed datasets exist under `data/processed/`.
3. **Training** (`src/models/trainer.py`):
   - Pulls the Iris dataset from scikit-learn.
   - Trains a RandomForest, evaluates accuracy, and logs metrics + model artefacts to MLflow (`mlflow.sklearn.log_model`).
   - Registers the model version under `MLFLOW_MODEL_NAME`, returning a resolvable `models:/...` URI for automation.
4. **Model registry**: MLflow stores versions, stages, and metrics. GitHub Actions consume the emitted `MODEL_URI` (and MLflow webhooks emit `repository_dispatch` events) to trigger deployments.
5. **Promotion**: During deploy workflows, the packaged image references the dispatched `MODEL_URI`, downloads the model during the Docker build, and is validated against live metrics before promotion.

## 4. Inference Service Architecture

- **Entrypoint**: `src/app/main.py` constructs the FastAPI app, wires routers (`health`, `predict`, `metrics`), and registers telemetry middleware.
- **Startup lifecycle**:
  - Reads `MODEL_PATH` and `LOG_LEVEL` from environment (see `src/app/config.py`).
  - Loads the model via `src/models/infer.load_model`, storing wrapper/metrics on `app.state`.
  - Seeds the `ml_model_accuracy` Prometheus gauge from `metrics.json`.
- **Request handling**:
  - `/health/`: reports readiness plus the last recorded metrics blob.
  - `/predict/`: validates payload shape, routes to the model wrapper, and normalises predictions.
  - `/metrics/`: serves Prometheus exposition format produced by `src/utils/telemetry.metrics_response`.
- **Telemetry**:
  - Middleware counts requests (`ml_request_count`), measures latency histograms, and tracks 5xx errors.
  - Accuracy gauge enables deploy-time gating and continuous monitoring.

## 5. CI/CD Automation (GitHub Actions)

- `ci-lint-test.yml`: multi-OS job running Ruff, MyPy, and pytest on pushes/PRs to `main`.
- `data-validation.yml`: on-demand or data changes; executes `python -m src.data.validators.cli --sample`.
- `model-training.yml`: trains the model against MLflow, registers a new version, and surfaces the resulting `MODEL_URI` as a workflow output.
- `deploy-canary-and-promote.yml`: consumes a `MODEL_URI` (via MLflow webhook or manual trigger), builds/pushes an image, deploys a Kubernetes canary via Helm, runs smoke tests, evaluates `ml_model_accuracy ≥ 0.70`, and either promotes or rolls back.

All workflows rely on Poetry-managed dependencies (Python 3.11). Secrets configure container registry credentials and kubeconfig data.

## 6. Infrastructure Footprint

- **Container image**: `docker/Dockerfile` installs requirements from `requirements.txt`, copies `src/`, downloads the specified MLflow model during build, and runs Uvicorn (`src.app.main:app`) on port 8000.
- **Helm chart** (`infra/helm/ml-model-chart`):
  - Deploys stable and optional canary deployments/services.
  - Ingress resources support weighted canary routing via Nginx annotations.
  - Values permit tuning replica counts, image repository/tag, and canary weight.
- **Monitoring** (`infra/monitoring`):
  - `ml-service-monitor.yaml`: Prometheus Operator `ServiceMonitor` for scraping `/metrics`.
  - `ml-recording-rules.yaml`: Recording rules for request error rate and p95 latency.
  - `test-inference-deployment.yaml`: Reference manifest bundling a minimal inference service used for monitoring tests/demos.

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

Keep this document in sync with architectural changes. Major updates (model lifecycle, deployment strategy, monitoring stack) should be reflected here alongside runbooks and policy documentation.
