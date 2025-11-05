# MLflow High Availability Configuration

## Overview

This document describes how to configure MLflow Tracking Server for high availability to eliminate single points of failure in the ML pipeline. A highly available MLflow setup ensures continuous operation of model training and deployment workflows even during infrastructure failures.

## Architecture Components

A production-grade MLflow HA setup consists of:

1. **Multiple MLflow Tracking Servers** - Load-balanced instances for redundancy
2. **PostgreSQL Backend Store** - Centralized metadata storage with replication
3. **S3-Compatible Artifact Store** - Distributed object storage for model artifacts
4. **Load Balancer** - Distributes traffic across MLflow instances
5. **Client Retry Logic** - Automatic failover handling in applications

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                            │
│                  (HAProxy / ALB / nginx)                     │
└─────────────┬──────────────┬──────────────┬─────────────────┘
              │              │              │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ MLflow  │    │ MLflow  │    │ MLflow  │
         │ Server  │    │ Server  │    │ Server  │
         │ Pod 1   │    │ Pod 2   │    │ Pod 3   │
         └────┬────┘    └────┬────┘    └────┬────┘
              │              │              │
              └──────────┬───┴──────────────┘
                         │
              ┌──────────▼───────────────┐
              │  PostgreSQL Backend      │
              │  (with replication)      │
              │  - Metadata storage      │
              │  - Run history           │
              │  - Model registry        │
              └──────────────────────────┘
                         │
              ┌──────────▼───────────────┐
              │  S3 / MinIO / GCS        │
              │  Artifact Storage        │
              │  - Model files           │
              │  - Metrics/logs          │
              │  - Dataset references    │
              └──────────────────────────┘
```

## Implementation Options

### Option 1: Self-Hosted MLflow on Kubernetes (Recommended for this project)

**Pros:**
- Full control over infrastructure
- Cost-effective for moderate scale
- Integrates well with existing K8s deployment

**Cons:**
- Requires operational overhead
- Need to manage PostgreSQL and S3 storage separately

#### Deployment Steps

1. **Deploy PostgreSQL with High Availability**

```yaml
# infra/helm/postgresql-ha-values.yaml
postgresql:
  replicaCount: 3
  persistence:
    enabled: true
    size: 50Gi
  metrics:
    enabled: true
  replication:
    enabled: true
    slaveReplicas: 2
    synchronousCommit: "on"
    numSynchronousReplicas: 1
```

Deploy using Bitnami PostgreSQL HA chart:
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install mlflow-postgres bitnami/postgresql-ha \
  -f infra/helm/postgresql-ha-values.yaml \
  --namespace mlflow --create-namespace
```

2. **Configure S3-Compatible Storage**

For AWS S3:
```bash
# Create S3 bucket with versioning
aws s3 mb s3://mlflow-artifacts-prod
aws s3api put-bucket-versioning \
  --bucket mlflow-artifacts-prod \
  --versioning-configuration Status=Enabled
```

For MinIO (self-hosted):
```yaml
# infra/helm/minio-values.yaml
replicas: 4
persistence:
  enabled: true
  size: 100Gi
mode: distributed
```

3. **Deploy MLflow Tracking Server**

Create `infra/helm/mlflow/values.yaml`:

```yaml
replicaCount: 3

image:
  repository: ghcr.io/mlflow/mlflow
  tag: "2.7.0"

service:
  type: ClusterIP
  port: 5000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: mlflow.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mlflow-tls
      hosts:
        - mlflow.example.com

env:
  - name: BACKEND_STORE_URI
    value: "postgresql://mlflow:password@mlflow-postgres-postgresql-ha-pgpool:5432/mlflowdb"
  - name: ARTIFACT_ROOT
    value: "s3://mlflow-artifacts-prod"
  - name: AWS_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: mlflow-s3-credentials
        key: access-key-id
  - name: AWS_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: mlflow-s3-credentials
        key: secret-access-key

command:
  - mlflow
  - server
  - --host
  - "0.0.0.0"
  - --port
  - "5000"
  - --backend-store-uri
  - $(BACKEND_STORE_URI)
  - --default-artifact-root
  - $(ARTIFACT_ROOT)
  - --gunicorn-opts
  - "--workers=4 --timeout=180"

resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

readinessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 30

podDisruptionBudget:
  enabled: true
  minAvailable: 2

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

Deploy MLflow:
```bash
kubectl create namespace mlflow
kubectl create secret generic mlflow-s3-credentials \
  --from-literal=access-key-id=$AWS_ACCESS_KEY_ID \
  --from-literal=secret-access-key=$AWS_SECRET_ACCESS_KEY \
  -n mlflow

helm install mlflow ./infra/helm/mlflow \
  -n mlflow
```

4. **Verify High Availability**

```bash
# Check pod distribution across nodes
kubectl get pods -n mlflow -o wide

# Test failover by deleting a pod
kubectl delete pod mlflow-0 -n mlflow

# Verify automatic recovery
kubectl get pods -n mlflow -w
```

### Option 2: Managed MLflow Services

#### AWS SageMaker Model Registry

**Pros:**
- Fully managed, no infrastructure overhead
- Native AWS integration
- Automatic scaling and backups

**Cons:**
- Vendor lock-in
- Higher cost at scale
- Limited customization

**Configuration:**
```python
import sagemaker
from sagemaker.model_registry import ModelPackageGroup

# Create model package group
mpg = ModelPackageGroup(
    model_package_group_name="iris-models",
    tags=[{"Key": "Environment", "Value": "Production"}]
)
```

#### Databricks Managed MLflow

**Pros:**
- Enterprise-grade HA out of the box
- Unified analytics platform
- Advanced governance features

**Cons:**
- Requires Databricks subscription
- Migration effort from self-hosted
- Higher cost

**Configuration:**
```python
import mlflow

# Configure Databricks MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/iris-experiment")
```

## Client Configuration for High Availability

### Environment Variables

Update application configuration to use HA-enabled MLflow:

```bash
# .env.production
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_ENABLE_RETRY=true
MLFLOW_RETRY_MAX_ATTEMPTS=5
MLFLOW_RETRY_BACKOFF_FACTOR=2
MLFLOW_CIRCUIT_BREAKER_THRESHOLD=10
MLFLOW_CIRCUIT_BREAKER_TIMEOUT=60
```

### Load Balancer Configuration

Example HAProxy configuration:

```haproxy
# /etc/haproxy/haproxy.cfg
frontend mlflow_frontend
    bind *:5000
    default_backend mlflow_backend

backend mlflow_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200

    server mlflow1 mlflow-0.mlflow.svc.cluster.local:5000 check
    server mlflow2 mlflow-1.mlflow.svc.cluster.local:5000 check
    server mlflow3 mlflow-2.mlflow.svc.cluster.local:5000 check
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **MLflow Server Metrics:**
   - Response time (p50, p95, p99)
   - Error rate (4xx, 5xx responses)
   - Request throughput
   - Active connections

2. **Database Metrics:**
   - Connection pool usage
   - Query latency
   - Replication lag
   - Disk usage

3. **Storage Metrics:**
   - Artifact upload/download success rate
   - Storage capacity
   - API error rates

### Prometheus Monitoring

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'mlflow'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - mlflow
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: mlflow
```

### Alert Rules

```yaml
# mlflow-alerts.yaml
groups:
  - name: mlflow
    interval: 30s
    rules:
      - alert: MLflowHighErrorRate
        expr: rate(mlflow_http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "MLflow server error rate above 5%"

      - alert: MLflowAllPodsDown
        expr: up{job="mlflow"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "All MLflow pods are down"

      - alert: PostgreSQLReplicationLag
        expr: pg_replication_lag > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL replication lag exceeds 10 seconds"
```

## Backup and Restore Procedures

### PostgreSQL Backup

```bash
#!/bin/bash
# scripts/backup-mlflow-db.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="mlflow_db_backup_${TIMESTAMP}.sql"

# Backup PostgreSQL database
kubectl exec -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  pg_dump -U mlflow mlflowdb > "${BACKUP_FILE}"

# Upload to S3
aws s3 cp "${BACKUP_FILE}" "s3://mlflow-backups/${BACKUP_FILE}"

# Rotate old backups (keep last 30 days)
find . -name "mlflow_db_backup_*.sql" -mtime +30 -delete
```

Schedule with cron:
```cron
0 2 * * * /path/to/scripts/backup-mlflow-db.sh
```

### S3 Artifact Backup

Enable S3 versioning and cross-region replication:

```bash
# Enable versioning
aws s3api put-bucket-versioning \
  --bucket mlflow-artifacts-prod \
  --versioning-configuration Status=Enabled

# Configure cross-region replication
aws s3api put-bucket-replication \
  --bucket mlflow-artifacts-prod \
  --replication-configuration file://replication-config.json
```

### Restore Procedure

```bash
#!/bin/bash
# scripts/restore-mlflow-db.sh

BACKUP_FILE=$1

# Download from S3
aws s3 cp "s3://mlflow-backups/${BACKUP_FILE}" .

# Restore to PostgreSQL
kubectl exec -i -n mlflow mlflow-postgres-postgresql-ha-pgpool-0 -- \
  psql -U mlflow mlflowdb < "${BACKUP_FILE}"
```

## Testing HA Configuration

### Chaos Engineering Tests

```bash
# Test 1: Kill random MLflow pod
kubectl delete pod -n mlflow -l app=mlflow --field-selector=status.phase=Running --all

# Verify auto-recovery
kubectl get pods -n mlflow -w

# Test 2: Simulate network partition
kubectl exec -n mlflow mlflow-0 -- iptables -A INPUT -j DROP

# Test 3: Load test with failover
hey -n 10000 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"run_id":"test","key":"metric","value":0.95,"timestamp":1234567890}' \
  https://mlflow.example.com/api/2.0/mlflow/runs/log-metric
```

### Validation Checklist

- [ ] Multiple MLflow pods running on different nodes
- [ ] Pod Disruption Budget configured (minAvailable: 2)
- [ ] PostgreSQL replication configured and healthy
- [ ] S3 bucket versioning enabled
- [ ] Load balancer health checks passing
- [ ] Automatic pod recovery verified
- [ ] Client retry logic implemented
- [ ] Circuit breaker patterns in place
- [ ] Monitoring alerts configured
- [ ] Backup/restore procedures tested
- [ ] Disaster recovery runbook documented

## Cost Considerations

### Self-Hosted MLflow (Kubernetes)

**Monthly estimate for moderate scale:**
- 3x MLflow pods (2 vCPU, 4GB RAM each): ~$150
- PostgreSQL HA (3 replicas): ~$200
- S3 storage (1TB): ~$23
- Load balancer: ~$20
- **Total: ~$393/month**

### Managed Services

**AWS SageMaker:**
- Model Registry: Free tier, then ~$0.65/hour for training
- Storage: Standard S3 pricing

**Databricks:**
- Pricing varies by plan (contact sales)
- Typically $0.40-$0.99/DBU + compute costs

## Migration Path

For teams currently using single-instance MLflow:

1. **Phase 1: Add Retry Logic (Week 1)**
   - Implement client-side retry mechanisms
   - Deploy without infrastructure changes
   - Monitor and tune retry parameters

2. **Phase 2: PostgreSQL Migration (Week 2)**
   - Backup existing MLflow database
   - Deploy PostgreSQL HA cluster
   - Migrate data from SQLite/single PostgreSQL
   - Update MLflow configuration

3. **Phase 3: Multi-Pod Deployment (Week 3)**
   - Scale MLflow to 3 replicas
   - Configure load balancer
   - Test failover scenarios

4. **Phase 4: Monitoring and Optimization (Week 4)**
   - Set up comprehensive monitoring
   - Configure alerts
   - Document procedures
   - Train team on runbooks

## References

- [MLflow Tracking Server Documentation](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers)
- [MLflow Database Schema](https://mlflow.org/docs/latest/tracking.html#backend-stores)
- [Kubernetes Pod Disruption Budgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
- [PostgreSQL High Availability](https://www.postgresql.org/docs/current/high-availability.html)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)

## Support

For issues or questions:
- Internal: Contact ML Platform Team (#ml-platform on Slack)
- MLflow Community: https://github.com/mlflow/mlflow/issues
- This Repository: https://github.com/arec1b0/ml-cicd-pipeline/issues
