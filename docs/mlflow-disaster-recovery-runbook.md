# MLflow Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for recovering from MLflow service failures and data loss scenarios. It covers common failure modes, recovery procedures, and verification steps.

**Target Audience:** DevOps, SRE, ML Engineers

**Last Updated:** 2025-11-05

---

## Table of Contents

1. [Emergency Contacts](#emergency-contacts)
2. [Failure Scenarios](#failure-scenarios)
3. [Recovery Procedures](#recovery-procedures)
4. [Verification Steps](#verification-steps)
5. [Post-Incident Actions](#post-incident-actions)

---

## Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| ML Platform Lead | [Contact Info] | 24/7 |
| DevOps On-Call | [Contact Info] | 24/7 |
| Database Admin | [Contact Info] | Business hours |
| Cloud Provider Support | [Support Portal] | 24/7 |

---

## Failure Scenarios

### Scenario 1: MLflow Server Unresponsive

**Symptoms:**
- Health check endpoints return 5xx errors
- Training jobs fail with connection timeout
- Model deployment workflows stuck
- Circuit breaker in OPEN state

**Impact:** HIGH - All ML workflows blocked

**Recovery Time Objective (RTO):** 15 minutes

**Recovery Point Objective (RPO):** 0 (no data loss)

---

### Scenario 2: PostgreSQL Database Failure

**Symptoms:**
- MLflow returns "database connection error"
- Cannot query model registry
- Run history unavailable
- Replication lag alerts firing

**Impact:** CRITICAL - Complete service outage

**RTO:** 30 minutes

**RPO:** 5 minutes (last backup)

---

### Scenario 3: S3 Artifact Storage Unavailable

**Symptoms:**
- Model downloads fail
- Artifact upload errors in training jobs
- "Access Denied" or timeout errors

**Impact:** HIGH - Training completes but models inaccessible

**RTO:** 20 minutes

**RPO:** 0 (S3 versioning enabled)

---

### Scenario 4: Complete MLflow Infrastructure Loss

**Symptoms:**
- All MLflow pods terminated
- Database and storage accessible but no API
- Kubernetes namespace deleted

**Impact:** CRITICAL - Complete rebuild required

**RTO:** 60 minutes

**RPO:** Based on last backup (target: 24 hours)

---

## Recovery Procedures

### Procedure 1: Restart MLflow Server Pods

**When to use:** MLflow pods crashed or unresponsive

**Prerequisites:**
- kubectl access to cluster
- Proper RBAC permissions

**Steps:**

```bash
# 1. Check pod status
kubectl get pods -n mlflow -l app=mlflow

# 2. Check pod logs for errors
kubectl logs -n mlflow mlflow-0 --tail=100

# 3. Restart all MLflow pods
kubectl rollout restart deployment/mlflow -n mlflow

# 4. Watch pod recovery
kubectl get pods -n mlflow -w

# 5. Verify health after restart (wait 30s for startup)
kubectl exec -n mlflow mlflow-0 -- curl -f http://localhost:5000/health
```

**Expected outcome:** All pods Running, health checks passing

**If unsuccessful:** Proceed to Procedure 2

---

### Procedure 2: Restore PostgreSQL Database

**When to use:** Database corruption, accidental deletion, or complete failure

**Prerequisites:**
- Access to S3 backup bucket
- PostgreSQL admin credentials
- Database connection string

**Steps:**

```bash
# 1. Identify latest valid backup
aws s3 ls s3://mlflow-backups/ --recursive | grep mlflow_db_backup | sort | tail -n 5

# 2. Download backup file
BACKUP_FILE="mlflow_db_backup_20251105_020000.sql"
aws s3 cp "s3://mlflow-backups/${BACKUP_FILE}" ./

# 3. Stop MLflow servers to prevent writes during restore
kubectl scale deployment/mlflow -n mlflow --replicas=0

# 4. Create maintenance window notification
# [Post to status page / notify team]

# 5. Drop and recreate database (CAREFUL!)
kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U postgres -c "DROP DATABASE mlflowdb;"

kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U postgres -c "CREATE DATABASE mlflowdb OWNER mlflow;"

# 6. Restore from backup
kubectl exec -i -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U mlflow mlflowdb < "${BACKUP_FILE}"

# 7. Verify database integrity
kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U mlflow mlflowdb -c "\dt"

kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U mlflow mlflowdb -c "SELECT COUNT(*) FROM runs;"

# 8. Restart MLflow servers
kubectl scale deployment/mlflow -n mlflow --replicas=3

# 9. Verify connectivity
kubectl exec -n mlflow mlflow-0 -- curl -f http://localhost:5000/health

# 10. Run smoke test (see Verification section)
```

**Expected outcome:** Database restored, MLflow operational

**Rollback:** Keep original backup, can re-attempt restore

---

### Procedure 3: Recover S3 Artifacts

**When to use:** Accidental deletion, corruption, or access issues

**Prerequisites:**
- S3 versioning enabled (should be by default)
- AWS CLI with appropriate permissions

**Steps:**

#### 3a. Restore Deleted Objects (if versioning enabled)

```bash
# 1. List deleted objects
aws s3api list-object-versions \
  --bucket mlflow-artifacts-prod \
  --prefix models/ \
  --query 'DeleteMarkers[?IsLatest==`true`].[Key,VersionId]' \
  --output text

# 2. Remove delete markers to restore objects
# For a specific object:
KEY="models/iris-random-forest/1/model.pkl"
DELETE_MARKER_ID="<version-id-from-above>"

aws s3api delete-object \
  --bucket mlflow-artifacts-prod \
  --key "${KEY}" \
  --version-id "${DELETE_MARKER_ID}"

# 3. Verify restoration
aws s3 ls "s3://mlflow-artifacts-prod/${KEY}"

# 4. For bulk restoration, use this script:
cat > restore_s3_objects.sh <<'EOF'
#!/bin/bash
BUCKET="mlflow-artifacts-prod"
PREFIX="models/"

aws s3api list-object-versions \
  --bucket "$BUCKET" \
  --prefix "$PREFIX" \
  --query 'DeleteMarkers[?IsLatest==`true`].[Key,VersionId]' \
  --output text | \
while read KEY VERSION_ID; do
  echo "Restoring: $KEY"
  aws s3api delete-object \
    --bucket "$BUCKET" \
    --key "$KEY" \
    --version-id "$VERSION_ID"
done
EOF

chmod +x restore_s3_objects.sh
./restore_s3_objects.sh
```

#### 3b. Fix S3 Permissions

```bash
# 1. Check bucket policy
aws s3api get-bucket-policy --bucket mlflow-artifacts-prod

# 2. Verify IAM role/user permissions
aws iam get-user-policy --user-name mlflow-service --policy-name MLflowS3Access

# 3. Re-apply correct policy if needed
aws s3api put-bucket-policy \
  --bucket mlflow-artifacts-prod \
  --policy file://s3-bucket-policy.json

# 4. Test access from MLflow pod
kubectl exec -n mlflow mlflow-0 -- \
  aws s3 ls s3://mlflow-artifacts-prod/models/ --region us-east-1
```

**Expected outcome:** Artifacts accessible, downloads succeed

---

### Procedure 4: Complete MLflow Rebuild

**When to use:** Infrastructure completely destroyed, namespace deleted, or migration to new cluster

**Prerequisites:**
- Helm charts available
- Database backup
- S3 bucket intact (or backup available)
- Configuration values documented

**Steps:**

```bash
# 1. Recreate namespace
kubectl create namespace mlflow

# 2. Create secrets
kubectl create secret generic mlflow-s3-credentials \
  --from-literal=access-key-id="${AWS_ACCESS_KEY_ID}" \
  --from-literal=secret-access-key="${AWS_SECRET_ACCESS_KEY}" \
  -n mlflow

kubectl create secret generic mlflow-db-credentials \
  --from-literal=username=mlflow \
  --from-literal=password="${DB_PASSWORD}" \
  -n mlflow

# 3. Deploy PostgreSQL (if not using external DB)
helm install mlflow-postgres bitnami/postgresql-ha \
  -f infra/helm/postgresql-ha-values.yaml \
  --namespace mlflow

# 4. Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql-ha \
  -n mlflow --timeout=300s

# 5. Restore database (if needed)
# Follow Procedure 2 steps 5-7

# 6. Deploy MLflow
helm install mlflow ./infra/helm/mlflow \
  --values ./infra/helm/mlflow/values.yaml \
  --namespace mlflow

# 7. Wait for MLflow pods
kubectl wait --for=condition=ready pod -l app=mlflow \
  -n mlflow --timeout=300s

# 8. Verify S3 connectivity
kubectl exec -n mlflow mlflow-0 -- \
  python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print(mlflow.list_experiments())"

# 9. Run full smoke test (see Verification section)

# 10. Update DNS / ingress if needed
kubectl get ingress -n mlflow
```

**Expected outcome:** Full MLflow stack operational

---

### Procedure 5: Reset Circuit Breaker

**When to use:** Circuit breaker stuck in OPEN state after issue resolved

**Prerequisites:**
- MLflow service confirmed healthy
- Root cause fixed

**Steps:**

```bash
# 1. Check circuit breaker state via health endpoint
curl https://mlflow.example.com/health | jq '.mlflow.circuit_breaker'

# 2. For inference service, use admin reload endpoint
curl -X POST https://ml-model-api.example.com/admin/reload

# 3. Verify circuit breaker reset
curl https://mlflow.example.com/health | jq '.mlflow.circuit_breaker.state'
# Should show: "closed"

# 4. Monitor for successful operations
kubectl logs -n ml-model -l app=ml-model --tail=100 | grep "Circuit breaker"
```

**Expected outcome:** Circuit breaker CLOSED, requests flowing

---

## Verification Steps

### Smoke Test Suite

Run after any recovery procedure:

```bash
#!/bin/bash
# smoke-test-mlflow.sh

set -e

MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow.mlflow.svc.cluster.local:5000}"

echo "=== MLflow Recovery Smoke Test ==="

# 1. Health check
echo "1. Testing health endpoint..."
curl -f "${MLFLOW_URI}/health" || exit 1

# 2. List experiments
echo "2. Listing experiments..."
python3 <<EOF
import mlflow
mlflow.set_tracking_uri("${MLFLOW_URI}")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")
assert len(experiments) > 0, "No experiments found"
EOF

# 3. Query model registry
echo "3. Querying model registry..."
python3 <<EOF
from mlflow import MlflowClient
client = MlflowClient("${MLFLOW_URI}")
models = client.search_registered_models()
print(f"Found {len(models)} registered models")
EOF

# 4. Download test artifact
echo "4. Testing artifact download..."
python3 <<EOF
import mlflow
mlflow.set_tracking_uri("${MLFLOW_URI}")
# Download latest production model
model_uri = "models:/iris-random-forest/Production"
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path="/tmp/test-download")
print("Artifact download successful")
EOF

# 5. Create test run
echo "5. Creating test run..."
python3 <<EOF
import mlflow
mlflow.set_tracking_uri("${MLFLOW_URI}")
mlflow.set_experiment("disaster-recovery-test")
with mlflow.start_run():
    mlflow.log_param("test", "recovery")
    mlflow.log_metric("status", 1.0)
print("Test run created successfully")
EOF

echo "=== All smoke tests passed ==="
```

### Performance Validation

```bash
# Check response times
echo "Testing MLflow performance..."

for i in {1..10}; do
  time curl -s "${MLFLOW_URI}/api/2.0/mlflow/experiments/search" > /dev/null
done

# Should complete in <500ms under normal conditions
```

### Data Integrity Checks

```sql
-- Connect to PostgreSQL
psql -U mlflow mlflowdb

-- Check for orphaned runs
SELECT COUNT(*) FROM runs WHERE experiment_id NOT IN (SELECT experiment_id FROM experiments);
-- Should return 0

-- Check for missing artifacts
SELECT run_id, artifact_uri FROM runs WHERE artifact_uri IS NULL LIMIT 10;
-- Should return few or no results

-- Verify recent activity
SELECT COUNT(*) FROM runs WHERE start_time > (EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000);
-- Should show recent runs if system was active

-- Check model registry consistency
SELECT COUNT(*) FROM model_versions WHERE run_id NOT IN (SELECT run_uuid FROM runs);
-- Should return 0
```

---

## Post-Incident Actions

### Immediate (Within 1 hour)

- [ ] Update status page with resolution
- [ ] Notify all stakeholders of recovery
- [ ] Document timeline and actions taken
- [ ] Check for any data loss or corruption
- [ ] Verify all dependent services recovered

### Short-term (Within 24 hours)

- [ ] Complete incident report
- [ ] Analyze root cause
- [ ] Review monitoring/alerting gaps
- [ ] Test backup/restore procedures
- [ ] Update runbook with lessons learned

### Long-term (Within 1 week)

- [ ] Implement preventive measures
- [ ] Enhance monitoring and alerting
- [ ] Schedule disaster recovery drill
- [ ] Update documentation
- [ ] Train team on new procedures

---

## Incident Report Template

```markdown
# MLflow Incident Report

**Incident ID:** [AUTO-GENERATED]
**Date:** [DATE]
**Severity:** [Critical/High/Medium/Low]
**Status:** [Resolved/Ongoing]

## Summary
[Brief description of the incident]

## Timeline
- **[TIME]:** Incident detected
- **[TIME]:** Investigation started
- **[TIME]:** Root cause identified
- **[TIME]:** Mitigation applied
- **[TIME]:** Service restored
- **[TIME]:** Verified recovery

## Impact
- **Users Affected:** [NUMBER/PERCENTAGE]
- **Services Down:** [LIST]
- **Duration:** [HOURS:MINUTES]
- **Data Loss:** [YES/NO - DETAILS]

## Root Cause
[Detailed explanation of what caused the incident]

## Resolution
[Steps taken to resolve the issue]

## Lessons Learned
- [LESSON 1]
- [LESSON 2]

## Action Items
- [ ] [ACTION 1] - Assigned to: [OWNER] - Due: [DATE]
- [ ] [ACTION 2] - Assigned to: [OWNER] - Due: [DATE]

## Prevention Measures
[Steps to prevent recurrence]
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: "Connection refused" errors

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n mlflow

# Check service endpoints
kubectl get endpoints -n mlflow

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl -it --rm -- \
  curl -v http://mlflow.mlflow.svc.cluster.local:5000/health
```

**Resolution:**
- Verify pods are Running
- Check service selector matches pod labels
- Ensure network policies allow traffic

---

#### Issue: "Database connection pool exhausted"

**Diagnosis:**
```bash
# Check PostgreSQL connections
kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check MLflow worker processes
kubectl exec -n mlflow mlflow-0 -- ps aux | grep gunicorn
```

**Resolution:**
```bash
# Increase connection pool in PostgreSQL
kubectl edit configmap mlflow-postgres-config -n mlflow
# Set max_connections = 200

# Increase worker timeout in MLflow
kubectl edit deployment mlflow -n mlflow
# Add: --gunicorn-opts="--workers=8 --timeout=300"

kubectl rollout restart deployment/mlflow -n mlflow
```

---

#### Issue: "Circuit breaker preventing requests"

**Diagnosis:**
```bash
# Check circuit breaker state
curl https://ml-model-api.example.com/health | jq '.mlflow'
```

**Resolution:**
- Verify MLflow is actually healthy
- If healthy, trigger reload: `curl -X POST https://ml-model-api.example.com/admin/reload`
- Monitor logs for successful reconnection
- Circuit breaker will automatically reset after successful requests

---

## Testing Disaster Recovery

### Quarterly DR Drill

Schedule regular disaster recovery drills to ensure procedures remain effective:

```bash
# 1. Plan drill (announce to team)
# 2. Choose scenario (rotate through all 4 scenarios)
# 3. Execute recovery procedure
# 4. Measure RTO/RPO achieved
# 5. Document findings and update runbook
```

### Automated DR Tests

```yaml
# .github/workflows/dr-test.yml
name: Disaster Recovery Test
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:

jobs:
  test-backup-restore:
    runs-on: ubuntu-latest
    steps:
      - name: Create test backup
        run: |
          # Backup PostgreSQL
          # Upload to S3

      - name: Simulate failure
        run: |
          # Delete test database

      - name: Restore from backup
        run: |
          # Execute restore procedure

      - name: Verify restoration
        run: |
          # Run smoke tests

      - name: Notify on failure
        if: failure()
        run: |
          # Send alert to ops channel
```

---

## Monitoring and Alerts

### Key Metrics

Monitor these metrics to detect issues early:

1. **MLflow Server Health**
   - Endpoint: `/health`
   - Alert if: Fails for >2 minutes

2. **Database Connections**
   - Metric: `pg_stat_activity`
   - Alert if: >80% of max_connections

3. **Artifact Storage**
   - Metric: S3 API errors
   - Alert if: Error rate >5%

4. **Circuit Breaker State**
   - Metric: Circuit state changes
   - Alert if: Opens (CLOSED → OPEN transition)

5. **Replication Lag**
   - Metric: PostgreSQL replication lag
   - Alert if: >10 seconds

### Alert Configuration

See `docs/mlflow-high-availability.md` for detailed Prometheus alert rules.

---

## Additional Resources

- [MLflow High Availability Setup](./mlflow-high-availability.md)
- [MLflow Official Documentation](https://mlflow.org/docs/latest/tracking.html)
- [PostgreSQL Backup & Recovery](https://www.postgresql.org/docs/current/backup.html)
- [AWS S3 Disaster Recovery](https://docs.aws.amazon.com/AmazonS3/latest/userguide/disaster-recovery-resiliency.html)

---

## Appendix: Quick Reference Commands

```bash
# Check MLflow status
kubectl get pods -n mlflow -l app=mlflow

# View MLflow logs
kubectl logs -n mlflow mlflow-0 --tail=100 -f

# Restart MLflow
kubectl rollout restart deployment/mlflow -n mlflow

# Scale MLflow
kubectl scale deployment/mlflow -n mlflow --replicas=5

# Check circuit breaker
curl https://ml-model-api.example.com/health | jq '.mlflow.circuit_breaker'

# Force model reload
curl -X POST https://ml-model-api.example.com/admin/reload

# Database backup
kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  pg_dump -U mlflow mlflowdb > backup.sql

# List S3 artifacts
aws s3 ls s3://mlflow-artifacts-prod/models/ --recursive | head -20
```

---

**Document Version:** 1.0
**Last Reviewed:** 2025-11-05
**Next Review:** 2026-02-05
