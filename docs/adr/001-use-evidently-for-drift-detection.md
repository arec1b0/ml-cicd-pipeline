# ADR 001: Use Evidently for Data and Model Drift Detection

## Status
Accepted

## Context

The ML-CICD pipeline requires robust monitoring for detecting data drift and model performance degradation in production. Multiple drift detection tools are available, including custom solutions, Evidently, WhyLabs, and others.

## Decision

We have chosen **Evidently AI** as the primary tool for data and model drift detection.

## Rationale

1. **Open Source Foundation**: Evidently is open-source with an active community, reducing vendor lock-in and allowing customization.

2. **Comprehensive Drift Detection**: Evidently provides:
   - Data drift detection (statistical tests)
   - Model performance monitoring (comparison against reference dataset)
   - Feature statistics tracking
   - Target drift detection

3. **Integration with MLflow**: Evidently integrates well with MLflow for logging metrics and tracking experiment results.

4. **Production Ready**: Evidently can be containerized and deployed as a standalone monitoring service, as implemented in `docker/Dockerfile.drift-monitor`.

5. **Ease of Use**: Provides intuitive Python API and HTML reports for stakeholder communication.

## Alternatives Considered

- **Custom Solution**: Would require significant development effort and maintenance overhead.
- **WhyLabs**: Proprietary solution with potential vendor lock-in; excellent features but higher cost.
- **Great Expectations**: Primarily focused on data validation rather than drift detection; complementary tool.

## Consequences

- **Positive**:
  - Automatic detection of production data issues
  - Early warning of model performance degradation
  - Reduced need for manual monitoring
  - Community support and ongoing development

- **Negative**:
  - Dependency on external library (though open-source)
  - Requires configuration of reference datasets for drift comparison
  - Storage overhead for historical data and reports

## Implementation Notes

- Drift monitoring is deployed in the cluster via `infra/monitoring/drift-monitor-deployment.yaml`
- Configuration is stored in `infra/monitoring/drift-monitor-config.yaml`
- Metrics are exported to Prometheus for visualization in Grafana

## See Also

- `docs/MONITORING.md` - Monitoring setup and dashboards
- `infra/monitoring/drift-monitor-deployment.yaml` - Deployment configuration

