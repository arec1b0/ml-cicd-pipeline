# Architecture Review

**Date**: Generated from current codebase analysis  
**Project**: ML CI/CD Pipeline  
**Reviewer**: Automated analysis

## Executive Summary

This repository demonstrates a **production-ready ML CI/CD pipeline** with comprehensive automation, observability, and deployment practices. The architecture follows modern ML Ops principles with clear separation of concerns, cross-platform support, and robust quality gates.

### Overall Assessment

**Strengths**: ⭐⭐⭐⭐⭐
- Well-structured codebase with clear module boundaries
- Comprehensive CI/CD automation with quality gates
- Production-ready observability (Prometheus, OpenTelemetry)
- Cross-platform support (Windows/Linux)
- Model versioning and drift monitoring integration
- Canary deployment strategy with automated promotion

**Areas for Improvement**: 
- Some architectural inconsistencies noted below
- Documentation gaps in some components
- Potential optimization opportunities

---

## 1. System Architecture Overview

### Architecture Diagram (Current State)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Flow                                │
└─────────────────────────────────────────────────────────────────┘

Raw Data → Validators → Trainer → MLflow Registry → Container Build
                                                           ↓
                    Production ← Stable Deployment ← Canary ← Image Push
                                                           ↓
                    Observability ← Prometheus ← Metrics Endpoint
                                                           ↓
                    Drift Detection ← Evidently ← Log Aggregation
```

### Key Components

1. **Training Pipeline** (`src/models/trainer.py`)
2. **Inference Service** (`src/app/main.py`)
3. **Drift Monitoring Service** (`src/drift_monitoring/`)
4. **CI/CD Automation** (`.github/workflows/`)
5. **Infrastructure** (`infra/helm/`, `infra/monitoring/`)

---

## 2. Detailed Component Analysis

### 2.1 Training Pipeline (`src/models/trainer.py`)

**Strengths:**
- ✅ Clean separation of concerns with `TrainResult` dataclass
- ✅ MLflow integration with model versioning
- ✅ Reference dataset persistence for drift detection
- ✅ ONNX export for production optimization
- ✅ Comprehensive logging and error handling

**Issues Identified:**
- ⚠️ **Hardcoded model parameters**: `n_estimators=10` is hardcoded (line 75). Should be configurable via environment variables or config file.
- ⚠️ **Single model type**: Only RandomForestClassifier supported. Architecture could be more extensible.
- ⚠️ **No hyperparameter tuning**: Training pipeline doesn't include tuning capabilities.

**Recommendations:**
```python
# Suggested improvement:
n_estimators = int(os.getenv("MODEL_N_ESTIMATORS", "10"))
random_state = int(os.getenv("MODEL_RANDOM_STATE", "42"))
```

**Architecture Alignment**: ✅ Aligns with MLflow-centric model lifecycle

---

### 2.2 Inference Service (`src/app/main.py`)

**Strengths:**
- ✅ Clean FastAPI application factory pattern
- ✅ Proper lifecycle management (startup/shutdown hooks)
- ✅ Integration with OpenTelemetry tracing
- ✅ Structured JSON logging
- ✅ Graceful error handling for model loading failures
- ✅ State management via `app.state` (avoids global variables)

**Issues Identified:**
- ⚠️ **Model loading path complexity**: Default path `/app/model/model/model.pkl` (line 17 in `config.py`) reflects MLflow artifact structure but is brittle. Path resolution could be more robust.
- ⚠️ **No model reloading**: Once loaded, model cannot be reloaded without restart. Consider adding a `/reload` endpoint for zero-downtime updates.
- ⚠️ **Metrics file dependency**: Relies on `metrics.json` being co-located with model (line 89). Could fail if MLflow structure changes.

**Recommendations:**
1. Add model reload capability:
```python
@app.post("/admin/reload")
async def reload_model():
    # Reload model from MODEL_PATH
    # Useful for zero-downtime model updates
```

2. Improve path resolution:
```python
def find_model_path(base_path: Path) -> Optional[Path]:
    """Search for model.pkl or model.onnx in MLflow structure."""
    # Try multiple possible locations
```

#### Remediation Plan – Runtime Model Hot Reloads

- **Decouple model artefacts from the image**: replace the build-time `download_model.py` step with a runtime fetch that runs inside the container. At startup the service should pull the `Production` (or configured) version from MLflow into a writable cache such as `/var/cache/model/<version>/`. Keep the latest successful path in `app.state` and expose it through telemetry for traceability.
- **Introduce a `ModelManager` helper**: encapsulate download, validation, and swap logic behind a small class (e.g., `src/models/manager.py`) that holds a read/write lock. This allows the request path to serve predictions off the current wrapper while background jobs prepare the next version.
- **Secure reload endpoint**: implement `POST /admin/reload` that accepts an admin token (header or mutual TLS). The handler should spawn a background task that:
  1. Resolves the target model via `MlflowClient.get_latest_versions`.
  2. Downloads artefacts to a new cache folder.
  3. Validates signature/metrics (rejects downgrade if accuracy regresses beyond threshold).
  4. Atomically swaps `app.state.ml_wrapper` and updates Prometheus gauges.
- **Optional background polling**: support `MODEL_AUTO_REFRESH_SECONDS` to poll MLflow for new versions, using the same manager to apply changes without reboots.
- **Rollout/rollback hooks**: emit structured events (e.g., to Loki) on successful swaps and keep the previous wrapper for quick rollback if the new model fails smoke tests.
- **Configuration additions**: extend `src/app/config.py` with items such as `MODEL_SOURCE` (`"mlflow"` vs `"local"`), `MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_STAGE`, `MODEL_CACHE_DIR`, and `ADMIN_API_TOKEN`. This keeps local development (local files) working while production uses the runtime path.
- **Deployment adjustments**: ensure the Kubernetes Deployment mounts an `emptyDir` (or persistent volume if warm start needed) for the cache path and grants the container write permissions. Document the requirement in Helm chart values (`infra/helm/ml-model-chart/values.yaml`).
- **Docs & runbooks**: update `docs/ARCHITECTURE.md` and operational runbooks with the new flow: “promote model in MLflow → call `/admin/reload` (or wait for auto-refresh) → observe metrics.”
- **Acceptance alignment**: this decouples model rollout from container rebuilds, the secured `/admin/reload` path satisfies the runtime refresh requirement, and the documentation updates address operational readiness.
- **Stretch goal**: if traffic patterns require horizontal scale-out or multi-model routing, assess managed serving layers (MLflow Model Serving, TorchServe, or Seldon Core) with the same reload contract.

**Architecture Alignment**: ✅ Follows FastAPI best practices, proper middleware ordering

---

### 2.3 Model Loading (`src/models/infer.py`)

**Strengths:**
- ✅ **ONNX-first approach**: Prioritizes ONNX for production (better performance)
- ✅ Graceful fallback to sklearn `.pkl` files
- ✅ Clean `ModelWrapper` abstraction
- ✅ Proper type hints and error handling

**Issues Identified:**
- ⚠️ **Path resolution logic**: ONNX path resolution (lines 71-75) assumes specific MLflow directory structure. Could be more flexible.
- ⚠️ **No model validation**: No verification that loaded model matches expected input/output schema.
- ⚠️ **Single input name assumption**: `input_name` defaults to `"float_input"` (line 29). Should be dynamically detected (which it is, but error handling could be better).

**Recommendations:**
1. Add model schema validation:
```python
def validate_model_schema(wrapper: ModelWrapper, expected_features: int):
    """Verify model accepts expected feature count."""
```

2. Improve ONNX path discovery:
```python
def find_onnx_model(base_path: Path) -> Optional[Path]:
    """Search for ONNX model in common MLflow locations."""
```

**Architecture Alignment**: ✅ Good abstraction layer, production-ready fallback strategy

---

### 2.4 Telemetry & Observability (`src/utils/telemetry.py`)

**Strengths:**
- ✅ Comprehensive Prometheus metrics (counter, histogram, gauge)
- ✅ Proper middleware implementation
- ✅ Error tracking (5xx responses)
- ✅ Latency measurement

**Issues Identified:**
- ⚠️ **Path cardinality**: Using full path as label (line 53) can cause high cardinality issues. Should use route templates instead.
- ⚠️ **Exception handling**: Exceptions are caught and re-raised, but metrics are recorded. Ensure this doesn't mask errors.

**Recommendations:**
```python
# Use route template instead of full path
path_template = _normalize_path(request.url.path)  # "/predict/" not "/predict/123"
REQUEST_COUNT.labels(method=method, path=path_template, status=str(status_code)).inc()
```

**Architecture Alignment**: ✅ Standard Prometheus patterns, proper middleware integration

---

### 2.5 Drift Monitoring (`src/drift_monitoring/`)

**Strengths:**
- ✅ Separate service architecture (decoupled from inference)
- ✅ Evidently integration for drift detection
- ✅ Background evaluation loop
- ✅ Prometheus metrics export
- ✅ Configurable via environment variables

**Issues Identified:**
- ⚠️ **Loose coupling**: Drift monitoring is separate service but integration points aren't clearly documented. How does it consume prediction logs?
- ⚠️ **Data ingestion**: `emit_prediction_log` (in `src/utils/drift.py`) writes to stdout/logs. Requires external log aggregation (Loki) to feed drift monitor. This dependency should be more explicit.
- ⚠️ **Reference dataset location**: Must be manually configured. Could be auto-discovered from MLflow.

**Recommendations:**
1. Document integration points clearly:
   - Prediction logs → Loki → Drift Monitor ingestion
   - Reference dataset URI configuration
   - Evaluation interval tuning

2. Consider direct integration option:
```python
# Optional: Direct API call to drift monitor
# Alternative to log-based ingestion
await drift_monitor_client.record_prediction(features, predictions)
```

**Architecture Alignment**: ✅ Microservices pattern, but coupling could be clearer

---

### 2.6 API Endpoints (`src/app/api/`)

**Strengths:**
- ✅ RESTful design
- ✅ Proper input validation with Pydantic
- ✅ Background task for logging (non-blocking)
- ✅ OpenTelemetry span instrumentation
- ✅ Correlation ID support

**Issues Identified:**
- ⚠️ **Batch size limits**: No validation on batch size. Large batches could cause memory issues.
- ⚠️ **Feature dimension validation**: Validates non-empty but not specific dimension (e.g., Iris expects 4 features).
- ⚠️ **Error messages**: Some error messages could be more specific for debugging.

**Recommendations:**
```python
# Add batch size limit
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))
if len(payload.features) > MAX_BATCH_SIZE:
    raise HTTPException(400, f"Batch size exceeds {MAX_BATCH_SIZE}")

# Validate feature dimensions
EXPECTED_FEATURES = 4  # Or from model metadata
if len(payload.features[0]) != EXPECTED_FEATURES:
    raise HTTPException(400, f"Expected {EXPECTED_FEATURES} features")
```

**Architecture Alignment**: ✅ Follows FastAPI patterns, good observability integration

---

### 2.7 CI/CD Workflows (`.github/workflows/`)

**Strengths:**
- ✅ Multi-OS testing (Ubuntu + Windows)
- ✅ Comprehensive quality gates (lint, type, test)
- ✅ Automated canary deployment
- ✅ Metrics-based promotion gates
- ✅ Proper secret management

**Issues Identified:**
- ⚠️ **Canary observation window**: Fixed 10-second wait (line 121 in `deploy-canary-and-promote.yml`). Should be configurable and potentially longer for meaningful metrics.
- ⚠️ **Single metric gate**: Only checks `ml_model_accuracy`. Could include latency, error rate checks.
- ⚠️ **Rollback on failure**: Canary is uninstalled on failure, but no notification mechanism documented.

**Recommendations:**
1. Make observation window configurable:
```yaml
env:
  CANARY_OBSERVATION_SECONDS: 300  # 5 minutes
```

2. Add multiple metric gates:
```bash
# Check latency P95
LATENCY=$(curl -sS http://127.0.0.1:${OBSERVATION_PORT}/metrics | grep ml_request_latency_seconds)
# Check error rate
ERROR_RATE=$(...)
```

3. Add notification step on rollback:
```yaml
- name: Notify on rollback
  if: failure()
  uses: actions/github-script@v6
  with:
    script: |
      github.issues.create({
        title: "Canary deployment failed",
        body: "Model accuracy below threshold"
      })
```

**Architecture Alignment**: ✅ GitOps principles, proper quality gates

---

### 2.8 Infrastructure (`infra/helm/`, `infra/monitoring/`)

**Strengths:**
- ✅ Complete Helm chart with stable + canary deployments
- ✅ HPA (Horizontal Pod Autoscaler) configuration
- ✅ Pod Disruption Budgets
- ✅ Network Policies
- ✅ Prometheus ServiceMonitor integration
- ✅ Recording rules for alerting

**Issues Identified:**
- ⚠️ **Ingress configuration**: Ingress template exists but weighted canary routing relies on Nginx annotations. Should document required ingress controller.
- ⚠️ **Resource limits**: Default resources (100m CPU, 256Mi memory) may be conservative. Should be tuned based on load testing.
- ⚠️ **Network policies**: Egress rules for MLflow/Loki/Jaeger are disabled by default. Should be enabled in production.

**Recommendations:**
1. Document ingress requirements:
   - Nginx Ingress Controller with canary annotations
   - Alternative: Istio/Gloo for advanced traffic splitting

2. Add resource recommendations:
```yaml
# values-production.yaml
resources:
  requests:
    cpu: 200m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi
```

**Architecture Alignment**: ✅ Kubernetes best practices, production-ready configuration

---

### 2.9 Docker & Containerization (`docker/`)

**Strengths:**
- ✅ Multi-stage build considerations (could be added)
- ✅ Model download during build (build-time model embedding)
- ✅ Proper Python environment setup
- ✅ ONNX support included

**Issues Identified:**
- ⚠️ **Build-time model download**: Model is downloaded during `docker build`. This means:
  - Image is tied to specific model version
  - Cannot update model without rebuilding image
  - Alternative: Download model at runtime (more flexible but adds startup time)

- ⚠️ **No multi-stage build**: Could reduce final image size by using build stage.

**Recommendations:**
1. Consider runtime model download option:
```dockerfile
# Alternative: Download at container startup
CMD ["python", "-m", "src.app.download_and_start", "--model-uri", "${MODEL_URI}"]
```

2. Add multi-stage build:
```dockerfile
FROM python:3.11-slim as builder
# Install build dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
# ... rest of build
```

**Architecture Alignment**: ✅ Standard Docker practices, but model embedding strategy is opinionated

---

## 3. Cross-Cutting Concerns

### 3.1 Configuration Management

**Current State:**
- Environment variables via `src/app/config.py`
- No config file support
- Some hardcoded values

**Issues:**
- ⚠️ **Scattered configuration**: Config is spread across multiple files
- ⚠️ **No validation**: No schema validation for environment variables
- ⚠️ **Default values**: Some defaults are reasonable, others may not suit all environments

**Recommendations:**
1. Consider Pydantic Settings:
```python
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    model_path: Path = Path("/app/model/model/model.pkl")
    log_level: str = "INFO"
    max_batch_size: int = 1000
    
    class Config:
        env_prefix = ""  # Or "ML_"
```

### 3.2 Error Handling

**Current State:**
- Good exception handling in most places
- Proper HTTP status codes
- Logging with context

**Issues:**
- ⚠️ **Inconsistent error formats**: Some errors return plain strings, others return structured JSON
- ⚠️ **Error correlation**: Correlation IDs help, but error aggregation could be improved

**Recommendations:**
1. Standardize error responses:
```python
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    correlation_id: Optional[str]
    details: Optional[Dict[str, Any]]
```

### 3.3 Testing Strategy

**Current State:**
- Unit tests in `tests/unit/`
- Data pipeline tests
- CI integration

**Issues:**
- ⚠️ **No integration tests**: Missing end-to-end tests for API → Model → Response flow
- ⚠️ **No load tests**: No performance/load testing in CI
- ⚠️ **Mock strategy**: Should document how to mock MLflow/model loading in tests

**Recommendations:**
1. Add integration tests:
```python
# tests/integration/test_api_flow.py
async def test_predict_endpoint_integration():
    # Test full request → model → response flow
```

2. Add load testing:
```yaml
# .github/workflows/load-test.yml
- name: Run k6 load tests
  run: k6 run tests/load/predict.js
```

---

## 4. Architecture Strengths

### ✅ **Separation of Concerns**
- Clear module boundaries (app, models, utils, drift_monitoring)
- Single Responsibility Principle followed
- Dependency injection patterns used

### ✅ **Observability**
- Comprehensive metrics (Prometheus)
- Distributed tracing (OpenTelemetry)
- Structured logging (JSON)
- Correlation IDs for request tracking

### ✅ **Deployment Strategy**
- Canary deployments with automated promotion
- Metrics-based gating
- Rollback capability
- Zero-downtime updates (via canary)

### ✅ **ML Ops Best Practices**
- Model versioning (MLflow)
- Reference dataset persistence
- Drift monitoring integration
- Production telemetry

### ✅ **Cross-Platform Support**
- Windows + Linux compatibility
- Platform-specific scripts
- Portable Python code

---

## 5. Architecture Weaknesses & Risks

### ⚠️ **Model Update Strategy**
**Risk**: Model is embedded in Docker image at build time.  
**Impact**: Cannot update model without rebuilding and redeploying container.  
**Mitigation**: Consider runtime model download or model server pattern.

### ⚠️ **Single Point of Failure**
**Risk**: No mention of high availability for MLflow tracking server.  
**Impact**: If MLflow is down, model training and deployment fail.  
**Mitigation**: Document MLflow HA setup or consider managed MLflow.

### ⚠️ **Data Validation Gap**
**Risk**: `src/data/validators` module is referenced but implementation not visible.  
**Impact**: Data validation workflow may fail if validators are missing.  
**Mitigation**: Ensure validators are implemented or document required structure.

### ⚠️ **Documentation Gaps**
**Risk**: Some integration points (drift monitoring, log aggregation) are not fully documented.  
**Impact**: Onboarding and operations become harder.  
**Mitigation**: Expand `docs/ARCHITECTURE.md` with integration diagrams.

---

## 6. Recommendations Summary

### High Priority

1. **Make model parameters configurable** (training pipeline)
2. **Add model reload endpoint** (inference service)
3. **Normalize Prometheus path labels** (telemetry)
4. **Document drift monitoring integration** (architecture docs)
5. **Add batch size limits** (API endpoints)

### Medium Priority

1. **Add integration tests** (testing)
2. **Consider runtime model download** (Docker)
3. **Expand CI/CD metric gates** (canary promotion)
4. **Add error response standardization** (API)
5. **Document ingress requirements** (infrastructure)

### Low Priority

1. **Add hyperparameter tuning** (training)
2. **Multi-stage Docker builds** (optimization)
3. **Load testing in CI** (performance)
4. **Pydantic Settings** (configuration)

---

## 7. Architecture Patterns Used

### ✅ **Application Factory Pattern**
- `create_app()` function in `main.py`
- Enables testing and multiple app instances

### ✅ **Middleware Pattern**
- PrometheusMiddleware, CorrelationIDMiddleware
- Cross-cutting concerns handled cleanly

### ✅ **Wrapper Pattern**
- `ModelWrapper` abstracts ONNX/sklearn differences
- Stable API regardless of underlying model format

### ✅ **Microservices Pattern**
- Separate drift monitoring service
- Decoupled from inference service

### ✅ **GitOps Pattern**
- Infrastructure as code (Helm)
- CI/CD automation
- Version-controlled deployments

---

## 8. Compliance & Best Practices

### ✅ **Follows**
- FastAPI best practices
- Kubernetes best practices
- MLflow model lifecycle
- Prometheus instrumentation standards
- Docker best practices (mostly)

### ⚠️ **Could Improve**
- Error handling standardization
- Configuration management
- Testing coverage (integration tests)
- Documentation completeness

---

## 9. Conclusion

This is a **well-architected ML CI/CD pipeline** that demonstrates production-ready practices. The codebase is clean, follows modern patterns, and includes comprehensive automation and observability.

**Key Strengths:**
- Clear separation of concerns
- Comprehensive CI/CD automation
- Production-ready observability
- Modern deployment strategies

**Primary Areas for Improvement:**
- Configuration management (make more values configurable)
- Documentation (especially integration points)
- Testing (add integration and load tests)
- Model update strategy (consider runtime download)

**Overall Rating**: ⭐⭐⭐⭐ (4/5)

The architecture is solid and production-ready with minor improvements recommended. The suggested changes would enhance maintainability, flexibility, and operational clarity.

---

## Appendix: Architecture Decision Records (ADRs)

Consider documenting key architectural decisions:

1. **ADR-001**: Why ONNX-first model loading?
2. **ADR-002**: Why build-time model embedding vs runtime download?
3. **ADR-003**: Why separate drift monitoring service?
4. **ADR-004**: Why canary deployment over blue-green?

These ADRs would help future maintainers understand design rationale.

