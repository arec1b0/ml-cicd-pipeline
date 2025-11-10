# Incident Response Runbook

This document provides procedures for responding to common incidents in the ML-CICD pipeline.

## Table of Contents

1. [Severity Levels](#severity-levels)
2. [General Response Procedure](#general-response-procedure)
3. [Model Performance Degradation](#model-performance-degradation)
4. [High Error Rate](#high-error-rate)
5. [High Latency](#high-latency)
6. [Data Drift Detection](#data-drift-detection)
7. [MLflow Service Down](#mlflow-service-down)
8. [Out of Memory Errors](#out-of-memory-errors)
9. [Communication Checklist](#communication-checklist)

---

## Severity Levels

| Level | Impact | Response Time | Escalation |
|-------|--------|----------------|------------|
| P1 (Critical) | Production outage, > 10% error rate | Immediate (< 5 min) | VP Engineering |
| P2 (High) | Partial degradation, 2-10% error rate | 15 minutes | Tech Lead |
| P3 (Medium) | Non-critical feature issue | 1 hour | On-call Engineer |
| P4 (Low) | Minor issues, no user impact | 24 hours | Ticket queue |

---

## General Response Procedure

### Phase 1: Initial Assessment (0-2 minutes)

1. **Acknowledge the alert**
   - Check incident notification in Slack/PagerDuty
   - Note: timestamp, affected service, severity level
   - Open Grafana to view current metrics

2. **Assess severity**
   - Error rate: Check `/metrics` endpoint or Grafana
   - User impact: Determine if production traffic affected
   - Scope: Single model or entire service?

3. **Notify stakeholders**
   - Critical (P1): Page on-call VP Engineering
   - High (P2): Message #ml-ops-oncall Slack channel
   - Include: Service name, current status, severity level

### Phase 2: Investigation (2-10 minutes)

1. **Gather context**
   - Check recent deployments in deployment history
   - Review error logs in Loki or Elasticsearch
   - Check resource usage (CPU, memory) in Grafana
   - Review model metrics and drift monitoring

2. **Check dependencies**
   - Is MLflow responding? `curl -I http://mlflow:5000`
   - Is database available? Check Prometheus metrics
   - Are data pipelines running? Check airflow/Prefect UI
   - Are upstream services healthy?

3. **Create incident ticket**
   - Open ticket in incident tracking system
   - Title: Service name + issue description
   - Assign to on-call engineer
   - Link to Grafana dashboard and relevant logs

### Phase 3: Remediation (varies by incident type)

See specific incident procedures below.

### Phase 4: Recovery Validation (5-15 minutes)

1. **Verify resolution**
   - Check error rate has returned to normal (< 0.5%)
   - Verify latency is back within SLA (p95 < 1.0s)
   - Confirm no cascading failures in downstream services

2. **Monitor closely**
   - Watch metrics for 15 minutes after recovery
   - Enable increased alerting sensitivity temporarily
   - Have engineer on standby for immediate re-escalation

### Phase 5: Post-Incident (after stabilization)

1. **Document incident**
   - Timeline of events
   - Root cause analysis
   - Resolution steps taken
   - Prevention for future

2. **Schedule postmortem**
   - For P1/P2 incidents: within 24 hours
   - Include: On-call engineer, tech lead, product owner
   - Follow blameless postmortem principles

---

## Model Performance Degradation

**Symptoms**: Model accuracy drop, increased prediction errors, unexpected output values

### Quick Diagnosis

```bash
# Check model version currently serving
kubectl get deployment ml-model-api -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODEL_VERSION")].value}'

# Check when model was deployed
kubectl rollout history deployment/ml-model-api

# View model metrics in Grafana
# Dashboard: ML Model Performance
```

### Step 1: Verify Model Still Loads

```bash
# Check model loading logs
kubectl logs deployment/ml-model-api --tail=50 | grep "model"

# Expected output: "Model loaded successfully from mlflow://models:/iris-random-forest/Production"
# Error output: "Failed to load model" → Model is corrupted
```

### Step 2: Check Input Data Quality

```bash
# Check for data drift alert
# Grafana: Monitoring dashboard → Evidently data drift status

# If drift detected (score > 0.3):
# 1. Review recent data changes
# 2. Check upstream data pipeline for errors
# 3. Compare to training data distribution
```

### Step 3: Immediate Remediation

**Option A: Rollback to previous model version**

```bash
# If previous version was working:
kubectl set env deployment/ml-model-api MODEL_VERSION=iris-random-forest:staging

# Verify rollback
kubectl rollout status deployment/ml-model-api

# Monitor error rate for 5 minutes
```

**Option B: Investigate and fix**

```bash
# If new model is expected, investigate root cause:
# 1. Review training data used for new model
# 2. Check feature preprocessing logic
# 3. Verify no data leakage
# 4. Compare model predictions with validation set

# If fixable without retraining:
# - Patch API validation logic
# - Redeploy application (not model)

# If requires retraining:
# - Move new model to Staging
# - Retrain from scratch or fine-tune
# - Re-promote after validation
```

### Step 4: Root Cause Analysis

After stabilization, investigate:
- What changed? (data distribution, features, model training code)
- Why wasn't it caught? (testing gaps, validation issues)
- How to prevent? (add metrics monitoring, data validation)

---

## High Error Rate

**Symptoms**: > 5% prediction errors, frequent 5xx HTTP responses

### Severity Assessment

- P1 (> 10% errors): Immediate rollback
- P2 (5-10% errors): Investigate first, then decide
- P3 (< 5% errors): Monitor and investigate

### Immediate Actions (First 2 Minutes)

```bash
# 1. Check application logs for error pattern
kubectl logs -f deployment/ml-model-api --tail=100 | grep ERROR

# 2. Check if specific input pattern is causing errors
# Look for: NaN values, out-of-range values, wrong dimensions

# 3. Check resource availability
kubectl top pods -l app=ml-model-api

# If CPU/memory maxed: Scale up or reduce batch size
kubectl scale deployment ml-model-api --replicas=5
```

### Investigation Flowchart

```
High Error Rate Detected
├─ Error Type: Malformed Input (400)
│  └─ → Skip to "Malformed Input" section
├─ Error Type: Model Not Ready (503)
│  └─ → Skip to "MLflow Service Down" section
├─ Error Type: Prediction Failed (500)
│  └─ Possible Causes:
│     ├─ Out of memory (OOM) → Scale up or reduce batch size
│     ├─ Model loading error → Check MLflow connectivity
│     ├─ Invalid feature values → Check data pipeline
│     └─ Downstream service error → Check database/cache
└─ Error Type: Timeout (504)
   └─ → Skip to "High Latency" section
```

### Remediation by Error Type

**Malformed Input (400 errors)**
- Normal behavior; API correctly rejects bad input
- If increasing, check upstream data sources

**Model Not Ready (503 errors)**
- Model failed to load
- → See "MLflow Service Down" section

**Prediction Failed (500 errors)**
- Check error message in logs
- If "Model not ready": Restart pods
- If "Out of memory": Scale deployment
- If "Feature error": Check data pipeline

```bash
# For OOM errors:
kubectl set resources deployment/ml-model-api --limits=memory=2Gi --requests=memory=1Gi

# For feature errors:
# Check data quality in source system
# Run data validation script
python -m src.data.validators validate /path/to/data
```

---

## High Latency

**Symptoms**: Prediction requests taking > 1 second (p95)

### Quick Diagnosis

```bash
# Check current latency
curl -H "Accept: application/openmetrics-text" http://localhost:8000/metrics | grep ml_request_latency

# Check trace timeline in Grafana
# Traces → Select recent request → View spans
```

### Investigation Steps

1. **Check server resource usage**
   ```bash
   kubectl top pods -l app=ml-model-api
   # If CPU > 80% or memory > 90%: Scale up
   ```

2. **Check database/cache latency**
   ```bash
   # Query database directly
   # If slow: Scale database or optimize queries
   ```

3. **Check model inference time**
   ```bash
   # From logs, look for span duration: "model_inference" span
   # If > 500ms: Model is slow
   #   - Check model file size (may be on slow storage)
   #   - Check feature preprocessing time
   #   - Consider model quantization or compression
   ```

4. **Check network latency**
   ```bash
   # From traces, if significant time in network calls:
   #   - Check MLflow service latency
   #   - Check inter-pod communication
   ```

### Remediation

**Scale up deployment**
```bash
kubectl scale deployment ml-model-api --replicas=5
kubectl autoscaling set deployment ml-model-api --min=3 --max=10 --cpu-percent=70
```

**Reduce batch size**
```bash
# If processing large batches (> 100), reduce MAX_BATCH_SIZE
kubectl set env deployment/ml-model-api MAX_BATCH_SIZE=50
```

**Optimize model**
```bash
# If model inference is slow, consider:
# - Model quantization (float32 → float16)
# - ONNX conversion for faster inference
# - Feature caching to reduce preprocessing
```

---

## Data Drift Detection

**Symptoms**: Evidently drift alert fired, drift score > 0.3

### Alert Response Flowchart

```
Data Drift Alert
├─ Is production traffic normal? (Check request volume)
│  ├─ No (Low/high traffic) → Possibly false alarm, continue monitoring
│  └─ Yes (Normal traffic) → Proceed to investigation
├─ Is model accuracy affected?
│  ├─ No → Monitored drift, not actionable, log and continue
│  └─ Yes (Accuracy down > 5%) → Proceed to remediation
└─ What data source changed?
   ├─ Upstream data pipeline → Fix data source
   ├─ User input distribution → Expected, no action
   └─ Unknown → Investigate further
```

### Investigation Steps

1. **Identify which features drifted**
   ```bash
   # Check Grafana: Data Quality dashboard → Feature drift scores
   # Identify top 3 features with highest drift
   ```

2. **Check upstream data source**
   ```bash
   # Compare recent data to training data
   python -c "
   import pandas as pd
   train = pd.read_csv('data/processed/train.csv')
   recent = pd.read_csv('data/recent.csv')
   for col in train.columns:
       print(f'{col}: train mean={train[col].mean():.2f}, recent mean={recent[col].mean():.2f}')
   "
   ```

3. **Assess impact**
   - Is model accuracy declining?
   - Are predictions changing systematically?
   - Are users affected?

### Remediation

**If drift is expected** (e.g., seasonal change):
- Document expected drift
- Update training data distribution baseline
- Continue monitoring

**If drift is unexpected**:
- Check upstream data pipelines for errors
- If data source broken: Fix and clear historical data
- If features changed: Update feature definitions
- Retrain model on recent data

**If accuracy is impacted**:
```bash
# 1. Immediately evaluate model performance
python -m src.models.trainer evaluate --test-data data/recent.csv

# 2. If accuracy drop > 5%:
#    - Move current model to Staging
#    - Retrain on recent data
#    - Promote new model after validation

# 3. If accuracy acceptable:
#    - Update reference baseline for drift monitoring
#    - Continue monitoring
```

---

## MLflow Service Down

**Symptoms**: Model fails to load at startup, cannot check out model versions

### Quick Diagnosis

```bash
# Check MLflow service status
kubectl get pods -n mlflow
kubectl logs -n mlflow deployment/mlflow

# Try to connect to MLflow
curl -I http://mlflow:5000/health
```

### Impact Assessment

- **At startup**: Application fails to start → 503 Service Unavailable
- **At model reload**: Application continues with cached model
- **During promotion**: Cannot transition model versions

### Immediate Remediation

**Option 1: Restart MLflow**
```bash
kubectl rollout restart deployment/mlflow -n mlflow
kubectl rollout status deployment/mlflow -n mlflow
```

**Option 2: Check MLflow database**
```bash
# If using PostgreSQL backend
kubectl exec -it postgresql-0 -- psql -U mlflow

# Check database is responsive
SELECT version();
```

**Option 3: Use cached model**
```bash
# Application will continue with previously loaded model
# This is acceptable for up to 2 hours while MLflow is restored
```

### Extended Resolution (> 2 hours down)

See `docs/mlflow-disaster-recovery-runbook.md` for:
- Backup and restore procedures
- High availability setup
- RTO/RPO targets

---

## Out of Memory Errors

**Symptoms**: "MemoryError" in logs, pod OOMKilled (code 137)

### Immediate Actions

```bash
# 1. Scale deployment to distribute load
kubectl scale deployment ml-model-api --replicas=1
# (temporarily reduce to allow failed pod to be recreated)

# Then scale back up after stabilization:
kubectl scale deployment ml-model-api --replicas=3

# 2. Check what's consuming memory
kubectl exec -it <pod-name> -- top -p <process-id>

# 3. Check resource limits
kubectl get deployment ml-model-api -o yaml | grep -A5 resources:
```

### Investigation

1. **Model size issue**
   ```bash
   # Check model file size
   du -sh mlflow/models/iris-random-forest/*/artifacts/model/
   
   # If > 1GB:
   #   - Consider model quantization
   #   - Use ONNX format (typically 30-50% smaller)
   #   - Check for unnecessary artifacts in model
   ```

2. **Feature preprocessing issue**
   ```bash
   # If processing large batches:
   #   - Reduce MAX_BATCH_SIZE
   #   - Check for data duplication in preprocessing
   #   - Profile memory usage with memory_profiler
   ```

3. **Memory leak**
   ```bash
   # If memory grows over time:
   #   - Check for cached objects not being garbage collected
   #   - Look for open file handles or database connections
   #   - Profile with py-spy: py-spy record -o profile.svg -- python app.py
   ```

### Remediation

**Short term**:
```bash
# Increase pod memory requests/limits
kubectl set resources deployment/ml-model-api \
  --requests=memory=1Gi \
  --limits=memory=2Gi
```

**Long term**:
- Optimize model size
- Implement streaming for large batches
- Add memory monitoring to alerting

---

## Communication Checklist

### Internal Communication

- [ ] Posted incident start time to `#ml-ops-oncall` Slack channel
- [ ] Included severity level, affected service, current status
- [ ] Posted updates every 10 minutes during active incident
- [ ] Posted resolution notification and post-incident ticket link

### External Communication

- [ ] Checked if customer-facing service affected
- [ ] If yes: Posted to status page (status.company.com)
- [ ] Sent email to customers affected (for P1 only)
- [ ] Posted post-incident summary in customer communication

### Post-Incident

- [ ] Created incident ticket with timeline
- [ ] Scheduled postmortem meeting (24 hours)
- [ ] Added prevention items to backlog
- [ ] Closed incident ticket when postmortem complete

---

## Escalation Matrix

### On-Call Contacts

| Role | Contact | Threshold |
|------|---------|-----------|
| ML Ops Engineer | `@oncall-ml` | All incidents |
| Tech Lead | `@ml-tech-lead` | P1 + P2 |
| VP Engineering | `@vp-eng` | P1 only |
| Database Admin | `@dba-oncall` | Data-related |

### Escalation Decision Tree

```
P1 (Critical)
├─ Page on-call ML Ops Engineer
├─ Page tech lead after 5 min if not improving
└─ Page VP Engineering after 15 min if not resolved

P2 (High)
├─ Message ML Ops Engineer in Slack
├─ If no response in 5 min: Page on-call
└─ Update tech lead every 15 minutes

P3 (Medium)
├─ Create Jira ticket
├─ Assign to available engineer
└─ No immediate page required
```

---

## Useful Commands Reference

```bash
# Check application status
kubectl get deployment ml-model-api
kubectl get pods -l app=ml-model-api
kubectl describe pod <pod-name>

# View logs
kubectl logs <pod-name> --tail=100
kubectl logs <pod-name> -f  # follow logs

# Check metrics
kubectl exec ml-model-api-0 -- curl localhost:8000/metrics | grep ml_

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/bash

# Rollback deployment
kubectl rollout undo deployment/ml-model-api
kubectl rollout history deployment/ml-model-api

# Scale deployment
kubectl scale deployment ml-model-api --replicas=5

# Update environment variable
kubectl set env deployment/ml-model-api MODEL_VERSION=iris-random-forest:staging
```

---

## See Also

- `docs/MONITORING.md` - Metrics and alerting
- `docs/MODEL_LIFECYCLE.md` - Model versioning
- `docs/mlflow-disaster-recovery-runbook.md` - MLflow recovery
- Grafana dashboards for real-time monitoring
- PagerDuty for on-call rotation

