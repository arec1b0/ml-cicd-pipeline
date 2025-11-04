# ml-cicd-pipeline

This repository provides an enterprise-ready CI/CD blueprint for machine learning services. It demonstrates how to move a model from experimentation to production with reproducible assets, automated quality gates, and observable runtime behavior across both Windows and Linux environments.

## Features

- **End-to-End Workflow:** Covers data validation, training, packaging, deployment, and post-deployment gating in a single, cohesive project.
- **Platform Agnostic:** Offers first-class support for Windows (via PowerShell) and Linux/macOS, plus containerized delivery.
- **Production Guardrails:** Implements GitHub Actions pipelines to enforce linting, type checking, testing, data validation, model metric thresholds, and canary promotion rules.
- **Operational Clarity:** Includes runbooks, policy documents, and monitoring manifests that reflect the real-world challenges of ML operations.

## Quick Start

### Prerequisites

- Python 3.11
- Poetry 1.5+ (or `pip` with `requirements.txt`)
- Docker, `kubectl`, and `helm` for containerized and Kubernetes workflows.

### Installation

<<<<<<< HEAD
### 3. Produce Artefacts & Run Service
1. Generate or copy datasets into `data/processed/` (satisfies validation tests).
2. Point `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_MODEL_NAME`, `MLFLOW_EXPERIMENT_NAME`) at your tracking server, then train and register a model: `poetry run python -m src.models.trainer`
   - Add `--output <path>` if you also want a local artefact for ad-hoc testing.
3. Start the API locally: `poetry run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000`
4. Validate via `GET /health/`, `POST /predict/`, and `GET /metrics/` (the health payload now includes an `mlflow` diagnostic block covering reachability and server version).

Containerised option: `MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_MODEL_NAME=iris-random-forest MLFLOW_MODEL_STAGE=Production MODEL_AUTO_REFRESH_SECONDS=300 ADMIN_API_TOKEN=dev-admin-token docker compose up --build` starts the API with runtime model downloads and exposes port `8000`—replace the token before exposing the deployment beyond local development. If your MLflow instance requires authentication, also pass `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` (or configure the Helm chart secrets).

## CI/CD Highlights

- **Lint, type, test** (`.github/workflows/ci-lint-test.yml`): Runs Ruff, MyPy, and pytest across Linux and Windows runners.
- **Data validation** (`data-validation.yml`): Executes validators when data assets change to catch schema drifts early.
- **Model training** (`model-training.yml`): Automates model retraining against MLflow, registers a new version, and surfaces the resulting `MODEL_URI` for downstream automation.
- **Canary deploy & promote** (`deploy-canary-and-promote.yml`): Reacts to MLflow `repository_dispatch` events (or manual triggers), builds/pushes images, derives runtime configuration from the supplied `MODEL_URI` (model name/stage), deploys a Helm canary, runs smoke tests, evaluates `ml_model_accuracy ≥ 0.70`, and promotes or rolls back accordingly.
=======
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ml-cicd-pipeline.git
    cd ml-cicd-pipeline
    ```

2.  **Install dependencies:**
    -   **Windows:**
        ```powershell
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
        .\scripts\windows\setup.ps1
        ```
    -   **macOS/Linux:**
        ```bash
        pip install poetry
        poetry install
        ```

### Usage

1.  **Train a model:**
    ```bash
    poetry run python -m src.models.trainer --output model.pkl
    ```

2.  **Run the API locally:**
    ```bash
    poetry run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
    ```

3.  **Test the API:**
    -   **Health Check:** `GET http://localhost:8000/health/`
    -   **Prediction:** `POST http://localhost:8000/predict/` with a JSON body like `{"features": [[1, 2, 3, 4]]}`
    -   **Metrics:** `GET http://localhost:8000/metrics/`

## Project Structure

-   `ci/`: Policy guidance and runbooks for the automated workflows.
-   `docs/`: In-depth documentation, including architecture and setup guides.
-   `infra/`: Infrastructure-as-code, including Helm charts for Kubernetes deployments and Prometheus monitoring manifests.
-   `notebooks/`: Jupyter notebooks for experimentation and analysis.
-   `scripts/`: Automation scripts for CI, Windows setup, and canary promotions.
-   `src/`: The main source code for the application.
    -   `app/`: The FastAPI application, including routers, configuration, and the main entry point.
    -   `models/`: Scripts for model training and inference.
    -   `utils/`: Utility modules for telemetry, logging, drift detection, and tracing.
-   `tests/`: Unit and integration tests for the application.
-   `.github/`: GitHub Actions workflow definitions.

## CI/CD Highlights

-   **`ci-lint-test.yml`:** Lints, type-checks, and tests the codebase on both Linux and Windows runners.
-   **`data-validation.yml`:** Runs data validators to detect schema drift when data assets change.
-   **`model-training.yml`:** Automates model retraining and registration in MLflow.
-   **`deploy-canary-and-promote.yml`:** Deploys a canary release, runs smoke tests, and promotes or rolls back the release based on performance metrics.
>>>>>>> 61c46b32f946dbaf77506e97a46a5133ab2ba0e1

## Drift Monitoring

This project includes a comprehensive drift monitoring and feedback loop system:

-   **Reference Snapshot:** The training script (`src/models/trainer.py`) persists the training dataset to a specified location. This snapshot serves as a baseline for drift detection.
-   **Production Telemetry:** The `/predict` endpoint emits structured JSON logs containing features, predictions, and metadata. These logs can be shipped to a log analysis service (e.g., Loki) to create a "current" dataset for drift analysis.
-   **Evidently Service:** A dedicated FastAPI service (`src/drift_monitoring/`) periodically compares the reference and production datasets using Evidently. It exports Prometheus gauges that track data drift, feature-level drift, and prediction PSI.
-   **Alerting:** The Prometheus monitoring setup includes recording rules that trigger alerts (`DataDriftDetected`, `HighPredictionPSI`) when significant drift is detected.

## Further Reading

-   **`docs/ARCHITECTURE.md`:** A detailed overview of the project's architecture and component responsibilities.
-   **`docs/SET-UP.md`:** Platform-specific setup instructions, environment configuration, and deployment guides.
-   **`API_DOCUMENTATION.md`:** A complete catalog of the API endpoints and their usage.
