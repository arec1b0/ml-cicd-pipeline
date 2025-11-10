# Model Lifecycle and Versioning Strategy

This document defines the model versioning strategy, lifecycle stages, promotion/demotion criteria, and procedures for managing model versions in production.

## Overview

Models progress through well-defined stages in MLflow Model Registry, enabling controlled rollouts, A/B testing, and safe rollback procedures.

## Model Stages

### 1. Development

- **Purpose**: Initial training and experimentation
- **Storage**: Local MLflow tracking server or artifact storage
- **Promotion Criteria**:
  - Passes unit tests and integration tests
  - Meets minimum performance benchmarks
  - Code review completed
  - Ready for staging evaluation
- **Duration**: Variable (days to weeks)

### 2. Staging

- **Purpose**: Pre-production validation and A/B testing
- **Environment**: Staging cluster with production-like data
- **Deployed Via**: Canary deployment (10-20% traffic)
- **Promotion Criteria**:
  - Passes staging tests for 24+ hours
  - No performance regression vs current production model
  - Data drift detection (if any) is understood and approved
  - Load testing passes (minimum 100 req/sec)
  - Security scanning passes
  - Product/stakeholder approval
- **Duration**: 24-48 hours
- **Rollback**: Automatic if error rate > 5% or latency p95 > 2.0s

### 3. Production

- **Purpose**: Serving predictions to end users
- **Environment**: Production cluster with 100% traffic
- **Deployment Strategies**:
  - **Blue-Green**: Immediate cutover after validation
  - **Canary**: 5% → 25% → 50% → 100% traffic over time
  - **Shadow**: New model runs in parallel, results logged but not used
- **Monitoring**: Continuous drift detection, performance monitoring, error tracking
- **Rollback**: Available within 5 minutes via previous version
- **Duration**: Until replaced by newer model or archived

### 4. Archived

- **Purpose**: Historical reference and compliance
- **Retention**: 90 days minimum (configurable per policy)
- **Access**: Read-only; not served to end users
- **Use Cases**: Audit trails, model comparison, reproduction of results
- **Cleanup**: Automated deletion after retention period expires

## Semantic Versioning

Models follow semantic versioning: `MAJOR.MINOR.PATCH`

```
1.2.3
│ │ └─ PATCH: Bug fixes, no feature changes
│ └─── MINOR: New optional features, backward compatible
└───── MAJOR: Breaking changes to input/output schema
```

### Examples

- `1.0.0`: Initial production model
- `1.0.1`: Bug fix in feature preprocessing
- `1.1.0`: New optional input feature added (backward compatible)
- `2.0.0`: Breaking change to output format or input schema

## Model Promotion Workflow

### Development → Staging

```bash
# 1. Review model metrics and explainability in Development
mlflow.tracking_uri = "http://localhost:5000"
model = mlflow.get_registered_model("iris-random-forest")

# 2. Transition to Staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="iris-random-forest",
    version=1,
    stage="Staging"
)

# 3. Kubernetes deployment updates canary deployment
# kubectl apply -f infra/helm/ml-model-chart/values-canary.yaml
```

### Staging → Production

```bash
# 1. After 24+ hour evaluation with positive results
# 2. Product approval obtained
# 3. Transition to Production
client.transition_model_version_stage(
    name="iris-random-forest",
    version=1,
    stage="Production"
)

# 4. Blue-Green deployment switches traffic (immediate or canary)
```

### Production → Archived

```bash
# 1. When replaced by newer model
# 2. After 30-day retention in Production
client.transition_model_version_stage(
    name="iris-random-forest",
    version=1,
    stage="Archived"
)

# 3. After 90 days in Archived, eligible for deletion
```

## Rollback Procedure

### Immediate Rollback (Emergency)

Use when production model experiences severe degradation (error rate > 10%, latency spike):

```bash
# 1. Identify previous good version
mlflow models list-versions --model-name iris-random-forest | grep Production

# 2. Transition current Production to Archived
client.transition_model_version_stage(
    name="iris-random-forest",
    version=2,  # Current broken version
    stage="Archived"
)

# 3. Transition previous version back to Production
client.transition_model_version_stage(
    name="iris-random-forest",
    version=1,  # Previous good version
    stage="Production"
)

# 4. Kubernetes deployment reloads model (no redeploy needed)
# Application will pick up new version at next startup or reload
```

### Graceful Rollback (Planned)

Use during normal model updates:

1. Transition new version from Production → Staging (if issues found)
2. Keep old version in Production temporarily
3. Investigate issues
4. Either fix and retry, or archive new version

## A/B Testing with Canary Deployments

The infrastructure supports running multiple model versions simultaneously for A/B testing.

### Setup

Two Kubernetes deployments exist:
- **Stable**: Runs current Production model
- **Canary**: Runs Staging or new Production model

Both serve traffic through the same ingress service via traffic splitting.

### Configuration

File: `infra/helm/ml-model-chart/values.yaml`

```yaml
stable:
  replicas: 3
  model_version: "iris-random-forest:1"
  traffic_weight: 95  # 95% of traffic

canary:
  replicas: 1
  model_version: "iris-random-forest:2"  # New version
  traffic_weight: 5   # 5% of traffic
```

### Procedure

1. **Deploy new model to canary**:
   ```bash
   helm upgrade ml-model-chart \
     --set canary.model_version="iris-random-forest:2" \
     --set canary.traffic_weight=5
   ```

2. **Monitor canary metrics**:
   - Error rate: Should match or be better than stable
   - Latency: Should be comparable
   - Data quality: Check for drift
   - Business metrics: Monitor prediction distributions

3. **Gradually shift traffic**:
   ```bash
   # Hour 1: 5% canary traffic (completed above)
   # Hour 2: 25% canary traffic
   helm upgrade ml-model-chart --set canary.traffic_weight=25
   
   # Hour 4: 50% canary traffic
   helm upgrade ml-model-chart --set canary.traffic_weight=50
   
   # Hour 8: 100% canary traffic (promote to stable)
   helm upgrade ml-model-chart --set stable.model_version="iris-random-forest:2" --set canary.traffic_weight=0
   ```

4. **Rollback if needed**:
   ```bash
   # Immediately redirect canary traffic back to stable
   helm upgrade ml-model-chart --set canary.traffic_weight=0
   ```

### Metrics to Monitor During A/B Test

| Metric | Threshold | Action |
|--------|-----------|--------|
| Error rate (canary - stable) | > 2% | Rollback canary |
| Latency p95 increase | > 20% | Investigate or rollback |
| Data drift score | > 0.3 | Halt, investigate data |
| Prediction PSI | > 0.2 | Review prediction distribution |
| Business metric change | > 5% | Analyze impact, may proceed |

## Model Archive and Retention

### Retention Policy

- **Production**: Keep until replaced (1-3 months typical)
- **Staging**: Keep until promoted to Production (7 days)
- **Archived**: Keep for 90 days (audit trail, compliance)

### Automatic Cleanup

Cron job (scheduled daily):
```bash
# Archive old versions not in Production/Staging for 90+ days
python -c "
import mlflow
from datetime import datetime, timedelta
client = mlflow.tracking.MlflowClient()
cutoff = datetime.now() - timedelta(days=90)
for mv in client.get_model_versions('iris-random-forest'):
    if mv.stage == 'Archived' and datetime.fromtimestamp(mv.last_updated_timestamp/1000) < cutoff:
        print(f'Deleting {mv.name} version {mv.version}')
        # Implementation: delete from artifact storage
"
```

## Model Metadata and Tagging

Each model version should include metadata for tracking and compliance:

```python
# During training
mlflow.set_tags({
    "environment": "production",
    "data_version": "2024-01-15",
    "feature_set": "v1.0.0",
    "training_date": "2024-01-15",
    "trainer_email": "data-scientist@company.com",
    "framework": "sklearn",
    "schema_version": "1.0.0"
})
```

### Recommended Tags

| Tag | Purpose | Example |
|-----|---------|---------|
| `schema_version` | Input/output schema version | `1.0.0` |
| `data_version` | Training data snapshot | `2024-01-15` |
| `feature_set_version` | Feature engineering version | `v1.2.0` |
| `training_date` | When model was trained | `2024-01-15` |
| `trainer_email` | Contact for questions | `ds@company.com` |
| `framework` | ML framework used | `sklearn`, `pytorch` |
| `explainability` | Explanation method | `shap`, `lime` |

## Performance Benchmarks

Models must meet these minimum criteria before promotion:

### Accuracy Metrics
- Classification: F1 score ≥ 0.85 (or domain-specific metric)
- Regression: RMSE ≤ baseline RMSE

### Latency Metrics
- p50 latency: ≤ 500ms
- p95 latency: ≤ 1.0s
- p99 latency: ≤ 2.0s

### Throughput
- Minimum: 100 predictions/sec per replica
- Expected: 500+ predictions/sec

### Error Handling
- Failed predictions: < 0.1%
- Malformed input rejection: 100% (no silent failures)

## Disaster Recovery

### If MLflow is Down

See `docs/mlflow-disaster-recovery-runbook.md` for detailed procedures.

**Quick Reference**:
- Models are cached in memory after loading
- Failed model loads prevent application startup (fail fast, not silent)
- Previous model versions available via snapshots
- Restore MLflow from backup within 2 hours

### If Production Model is Corrupted

1. Identify corruption immediately via error rate alerts
2. Trigger immediate rollback to previous version
3. Investigate corruption cause
4. Retrain model from clean data

## Compliance and Audit Trail

All model transitions are logged in MLflow with:
- Timestamp of change
- User who made the change
- Model version and stage
- Reason for change (optional annotation)

Access logs for compliance:
```bash
mlflow.tracking.MlflowClient().search_model_versions(filter_string="tags.environment='production'")
```

## Related Documents

- `docs/adr/002-runtime-model-loading.md` - Runtime model loading rationale
- `docs/adr/003-schema-versioning-strategy.md` - Schema versioning strategy
- `docs/MONITORING.md` - Monitoring and alerting
- `ci/runbooks/incident-response.md` - Incident response procedures
- `docs/mlflow-disaster-recovery-runbook.md` - MLflow HA and recovery

