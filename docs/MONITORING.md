# Monitoring and Observability Documentation

This document describes the monitoring infrastructure, dashboards, alerts, and observability stack for the ML-CICD pipeline.

## Observability Stack

The monitoring architecture uses the "Three Pillars of Observability":

### 1. Metrics (Prometheus + Grafana)

- **Prometheus**: Time-series database collecting metrics from all components
- **Grafana**: Visualization and dashboarding platform
- **ServiceMonitor**: Kubernetes custom resources defining scrape targets

### 2. Logs (Loki)

- **Loki**: Log aggregation system compatible with Prometheus ecosystem
- **Integration**: Structured JSON logging from application with correlation IDs
- **Querying**: LogQL language for log queries in Grafana

### 3. Traces (Tempo)

- **Tempo**: Distributed tracing backend
- **OpenTelemetry**: Standard instrumentation in application code
- **Correlation**: Traces linked to logs via trace IDs for end-to-end observability

## Key Metrics

### ML Model Metrics

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|-----------------|
| `ml_request_count` | Counter | Total prediction requests | - |
| `ml_request_errors_total` | Counter | Failed predictions | - |
| `ml_request_latency_seconds` | Histogram | Prediction latency | p95 > 1.0s |
| `job:ml_error_rate:ratio` | Recording Rule | Error rate (5-min avg) | > 5% |
| `job:ml_p95_latency_seconds` | Recording Rule | 95th percentile latency | > 1.0s |

### Data Quality Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `evidently_data_drift_status` | Drift Monitor | Data drift detected (0=no, 1=yes) |
| `evidently_prediction_psi_score` | Drift Monitor | Prediction stability index | 
| `evidently_feature_statistics` | Drift Monitor | Per-feature drift scores |

## Prometheus Recording Rules

The file `infra/monitoring/ml-recording-rules.yaml` defines:

```yaml
# Error rate over 5-minute window
job:ml_error_rate:ratio = sum(rate(ml_request_errors_total[5m])) / sum(rate(ml_request_count[5m]))

# 95th percentile latency (p95)
job:ml_p95_latency_seconds = histogram_quantile(0.95, sum(rate(ml_request_latency_seconds_bucket[5m])) by (le))
```

These pre-computed metrics improve dashboard performance and enable efficient alerting.

## Alerts

### Alert: Data Drift Detected
- **Expression**: `evidently_data_drift_status == 1`
- **Duration**: 10 minutes
- **Severity**: Warning
- **Action**: Review data distribution changes; consider retraining if drift is significant

### Alert: High Prediction PSI (Population Stability Index)
- **Expression**: `evidently_prediction_psi_score > 0.2`
- **Duration**: 15 minutes
- **Severity**: Critical
- **Action**: Investigate input data skew; trigger model retraining evaluation

## Grafana Dashboards

### Datasources

Grafana is configured with three datasources:

1. **Prometheus** (default)
   - URL: `http://prometheus:9090`
   - Scrape interval: 15 seconds
   - Used for: Metrics queries

2. **Loki**
   - URL: `http://loki:3100`
   - Max lines: 1000
   - Used for: Log queries and correlation to traces

3. **Tempo**
   - URL: `http://tempo:3200`
   - Features:
     - Trace-to-logs correlation via trace IDs
     - Logs-to-traces correlation
     - Service map visualization
     - Node graph for dependency analysis

Configuration: `infra/monitoring/grafana-datasources.yaml`

### Recommended Dashboards

#### 1. ML Model Performance Dashboard
- **Purpose**: Real-time model serving metrics
- **Panels**:
  - Request rate (req/sec)
  - Error rate (%)
  - Latency (p50, p95, p99)
  - Batch size distribution
  - Model version in-use

#### 2. Data Quality Dashboard
- **Purpose**: Monitor data drift and data quality
- **Panels**:
  - Evidently data drift status
  - Prediction PSI score over time
  - Feature-level drift scores
  - Input value distributions (histograms)

#### 3. System Health Dashboard
- **Purpose**: Infrastructure and resource monitoring
- **Panels**:
  - Pod CPU and memory usage
  - Network I/O
  - Disk usage
  - MLflow service availability

#### 4. Traces and Logs Dashboard
- **Purpose**: Debugging and performance analysis
- **Panels**:
  - Request trace waterfall
  - Error logs with traces
  - Latency breakdown by component
  - Correlation ID search

## ServiceMonitor Configuration

The `ml-model-servicemonitor` (in `infra/monitoring/ml-service-monitor.yaml`) configures Prometheus to scrape the ML model service:

- **Service**: `ml-model-svc` (default namespace)
- **Port**: `http` (port 8000)
- **Endpoint**: `/metrics`
- **Interval**: 15 seconds
- **Honor Labels**: True (respects metric label values from target)

### Updating ServiceMonitor

If your service has a different name or namespace, update the selector:

```yaml
selector:
  matchNames:
    - your-service-name
namespaceSelector:
  matchNames:
    - your-namespace
```

## Drift Monitor Deployment

The drift monitoring service runs as a separate deployment:

**Configuration**: `infra/monitoring/drift-monitor-config.yaml`
**Deployment**: `infra/monitoring/drift-monitor-deployment.yaml`
**Service**: `infra/monitoring/drift-monitor-service.yaml`
**ServiceMonitor**: `infra/monitoring/drift-monitor-servicemonitor.yaml`

### Features

- Compares production data against training data baseline
- Detects feature drift, model output drift, and target drift
- Exports metrics to Prometheus in Evidently format
- Generates HTML reports for manual inspection

### Configuration Example

```yaml
reference_data_path: /data/reference/train.csv
comparison_interval: 1h
metrics:
  - data_drift
  - prediction_drift
  - feature_statistics
```

## Logging and Structured Logging

### Log Format

All logs are structured JSON with standard fields:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "src.app.api.predict",
  "message": "Prediction completed successfully",
  "correlation_id": "abc123def456",
  "prediction_count": 5,
  "trace_id": "xyz789"
}
```

### Key Fields

- `correlation_id`: Request tracking across services
- `trace_id`: OpenTelemetry trace ID for distributed tracing
- `span_id`: OpenTelemetry span ID for detailed traces

### Querying Logs in Grafana

Example Loki query for failed predictions:

```
{job="ml-model-api"} | json | level="ERROR" | pattern "Prediction failed"
```

## Common Troubleshooting

### Issue: Metrics Not Appearing in Prometheus

1. Verify ServiceMonitor is created:
   ```bash
   kubectl get servicemonitor -n monitoring
   ```

2. Check Prometheus scrape targets:
   - Navigate to Prometheus UI (port 9090)
   - Status → Targets
   - Verify `ml-model-svc` is listed and healthy

3. Verify application `/metrics` endpoint:
   ```bash
   kubectl port-forward svc/ml-model-svc 8000:8000
   curl http://localhost:8000/metrics
   ```

### Issue: Drift Alerts Firing Constantly

1. Verify reference dataset is representative
2. Check drift threshold settings in `drift-monitor-config.yaml`
3. Review `evidently_data_drift_status` metric values in Prometheus

### Issue: Logs Not Appearing in Loki

1. Verify application is writing to stdout (JSON format)
2. Check Loki is running:
   ```bash
   kubectl get pods -n monitoring | grep loki
   ```
3. Verify log parser configuration in Loki

## Best Practices

1. **Set meaningful labels**: Use labels for model versions, deployments (stable/canary), environments
2. **Monitor SLIs**: Track Service Level Indicators (latency, error rate, availability)
3. **Set appropriate alert thresholds**: Avoid alert fatigue with realistic thresholds
4. **Document dashboard changes**: Update this file when adding new dashboards
5. **Test alerting**: Verify alerts fire and notify the on-call engineer
6. **Regular review**: Weekly review of alerts and dashboard effectiveness

## Related Documents

- `docs/MODEL_LIFECYCLE.md` - Model versioning and promotion strategy
- `ci/runbooks/incident-response.md` - Incident response procedures
- `infra/monitoring/` - All monitoring configuration files

