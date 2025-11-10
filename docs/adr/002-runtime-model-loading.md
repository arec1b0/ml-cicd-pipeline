# ADR 002: Runtime Model Loading Strategy

## Status
Accepted

## Context

The ML-CICD pipeline needs to deploy model updates without rebuilding container images. Two primary strategies exist:
1. **Build-time**: Bundle models into the container image during build
2. **Runtime**: Load models at startup from external storage (MLflow, S3, etc.)

## Decision

We have chosen **runtime model loading** from MLflow Model Registry.

## Rationale

1. **Deployment Velocity**: Models can be updated and promoted in MLflow without rebuilding or redeploying containers, enabling rapid iteration and rollback.

2. **Separation of Concerns**: 
   - Container image focuses on code and dependencies
   - Model artifacts are managed independently via MLflow
   - Enables independent scaling of model training and model serving

3. **Model Registry Integration**: MLflow Model Registry provides:
   - Version tracking and lifecycle management (Staging, Production, Archived)
   - Model comparison and metrics
   - Audit trails for regulatory compliance

4. **Efficient Resource Utilization**: 
   - Smaller container images (faster pulls, reduced storage)
   - Models cached on persistent storage shared across instances
   - No need to rebuild on every model update

5. **Canary Deployments**: Runtime loading enables serving multiple model versions simultaneously for A/B testing through Kubernetes deployments.

## Alternatives Considered

- **Build-time Model Bundling**: Simpler initial implementation but requires full pipeline rebuild for each model update.
- **Model Sidecar Pattern**: Separate containers for models, adds complexity without significant benefit over MLflow.

## Consequences

- **Positive**:
  - Fast model updates without container rebuilds
  - Easy rollback to previous model versions
  - A/B testing of multiple model versions
  - Cleaner separation of model and code lifecycle

- **Negative**:
  - Requires external storage system (MLflow) to be reliable
  - Network latency on model loading (mitigated by caching)
  - Requires configuration of MLflow credentials in deployment
  - Model loading failures are visible at runtime rather than build time

## Implementation Details

- Model loading occurs in `src/app/main.py` during application startup
- `src/models/manager.py` handles MLflow integration and model loading
- Models are cached in memory after initial load
- Failed model loading prevents application readiness (503 Service Unavailable)

## Disaster Recovery

- If MLflow is unavailable at startup, cached models can be used (see `src/resilient_mlflow.py`)
- Fallback to previous model versions available via MLflow version history
- See `docs/mlflow-disaster-recovery-runbook.md` for detailed procedures

## See Also

- `src/models/manager.py` - Model loading implementation
- `docs/MODEL_LIFECYCLE.md` - Model versioning and promotion strategy
- `docs/mlflow-disaster-recovery-runbook.md` - MLflow HA and recovery

