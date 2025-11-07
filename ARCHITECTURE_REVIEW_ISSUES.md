# Architecture Review - GitHub Issues

This document contains formulated GitHub issues based on a comprehensive architecture review conducted on 2025-11-07.

**Summary**: 68 issues identified across 12 categories, organized into 12 actionable GitHub issues below.

---

## 🔴 CRITICAL PRIORITY

### Issue 1: [SECURITY] Critical Security Vulnerabilities

**Priority**: Critical
**Labels**: security, bug, priority:critical

**Description**:
Multiple critical security vulnerabilities have been identified that need immediate attention:

**1. Hardcoded Default Admin Token**
- **Location**: `docker-compose.yml:18`
- **Issue**: Default admin token `dev-admin-token` is hardcoded
- **Risk**: Provides trivial authentication bypass if deployed without override

**2. Missing Input Validation on Environment Variables**
- **Location**: `src/app/config.py`
- **Issue**: No validation for max values on `MAX_BATCH_SIZE`, `MODEL_AUTO_REFRESH_SECONDS`
- **Risk**: DoS attack via resource exhaustion (e.g., `MAX_BATCH_SIZE=999999999`)

**3. No Rate Limiting**
- **Location**: API endpoints
- **Issue**: `/predict` endpoint has no rate limiting
- **Risk**: API abuse, resource exhaustion

**4. Loki Query Injection Potential**
- **Location**: `src/drift_monitoring/monitor.py:419`
- **Issue**: Loki query passed directly without sanitization
- **Risk**: LogQL injection if query is user-configurable

**5. No TLS Configuration**
- **Location**: Dockerfile exposes port 8000
- **Issue**: No HTTPS support in application
- **Risk**: Credentials and predictions transmitted in cleartext

**Acceptance Criteria**:
- [ ] Remove hardcoded token from docker-compose.yml, require explicit configuration
- [ ] Add validation for MAX_BATCH_SIZE (max: 10000) and MODEL_AUTO_REFRESH_SECONDS (max: 3600)
- [ ] Implement rate limiting middleware (e.g., 100 requests/minute per IP)
- [ ] Sanitize/validate Loki queries or use parameterized queries
- [ ] Add TLS configuration documentation and support at ingress level
- [ ] Add security scanning (bandit, safety) to CI/CD pipeline

**Affected Files**:
- `docker-compose.yml`
- `src/app/config.py`
- `src/app/main.py`
- `src/drift_monitoring/monitor.py`
- `.github/workflows/`

---

### Issue 2: [BUG] Feature Dimension Mismatch Causing Prediction Failures

**Priority**: Critical
**Labels**: bug, priority:critical, model

**Description**:
The `EXPECTED_FEATURE_DIMENSION` default value is set to 10, but the Iris dataset has only 4 features. This causes all predictions to fail with the default configuration.

**Location**: `src/app/api/predict.py:25`

**Current Code**:
```python
EXPECTED_FEATURE_DIMENSION = int(os.getenv("EXPECTED_FEATURE_DIMENSION", "10"))
```

**Issue**: The default value of 10 is incorrect for the Iris model which expects 4 features.

**Impact**: All prediction requests will be rejected with validation errors unless the environment variable is explicitly set.

**Acceptance Criteria**:
- [ ] Change default EXPECTED_FEATURE_DIMENSION to 4
- [ ] OR derive feature dimension from loaded model metadata
- [ ] Add validation to compare expected vs actual model input shape at startup
- [ ] Update documentation with correct default value
- [ ] Add test to verify feature dimension matches model expectations

**Affected Files**:
- `src/app/api/predict.py`
- `docker-compose.yml`
- `infra/helm/ml-model-chart/values.yaml`
- `README.md`

---

### Issue 3: [SECURITY] Weak Secret Management in Kubernetes

**Priority**: Critical
**Labels**: security, kubernetes, priority:critical

**Description**:
The Helm chart allows inline secret values in `values.yaml`, which encourages committing secrets to version control.

**Location**: `infra/helm/ml-model-chart/values.yaml:97-101`

**Current Configuration**:
```yaml
adminTokenSecretValue: ""   # when set, chart will generate a secret automatically
```

**Issues**:
1. Secrets may be committed to version control
2. No integration with external secret management
3. Insufficient RBAC - no ServiceAccount, Role, or RoleBinding defined
4. Pods run with default service account permissions

**Acceptance Criteria**:
- [ ] Remove `adminTokenSecretValue` from values.yaml
- [ ] Add support for Sealed Secrets or External Secrets Operator
- [ ] Create dedicated ServiceAccount with minimal permissions
- [ ] Define Role and RoleBinding for the service
- [ ] Add documentation for secret management best practices
- [ ] Update deployment runbook with secret rotation procedures

**Affected Files**:
- `infra/helm/ml-model-chart/values.yaml`
- `infra/helm/ml-model-chart/templates/deployment-*.yaml`
- `infra/helm/ml-model-chart/templates/serviceaccount.yaml` (new)
- `infra/helm/ml-model-chart/templates/role.yaml` (new)
- `ci/runbooks/deploy-runbook.md`

---

## 🟡 HIGH PRIORITY

### Issue 4: [CONFIG] Configuration Management and Validation Issues

**Priority**: High
**Labels**: configuration, technical-debt, priority:high

**Description**:
Multiple configuration management issues that can lead to runtime failures and difficult debugging:

**1. No Configuration Validation**
- **Location**: `src/app/config.py`
- **Issue**: No validation that required variables are set for chosen `MODEL_SOURCE`
- **Impact**: Runtime failures (e.g., `MODEL_SOURCE=mlflow` without `MLFLOW_MODEL_NAME`)

**2. Inconsistent Default Values**
- `MODEL_AUTO_REFRESH_SECONDS`: Default `0` (disabled) in config, `300` in docker-compose
- Feature dimension: `10` in predict.py, `4` in actual Iris dataset

**3. Missing Environment Variable Documentation**
- Many env vars undocumented: `MLFLOW_RETRY_MAX_ATTEMPTS`, `MLFLOW_CIRCUIT_BREAKER_THRESHOLD`, `EXPECTED_FEATURE_DIMENSION`

**4. Hardcoded Paths**
- **Location**: `src/app/config.py:36`
- `DEFAULT_MODEL_PATH = "/app/model/model/model.pkl"` assumes specific container structure

**Acceptance Criteria**:
- [ ] Migrate to Pydantic BaseSettings with validators
- [ ] Add validation for MODEL_SOURCE-specific required variables
- [ ] Centralize all default values in one location
- [ ] Create comprehensive environment variable reference document
- [ ] Make hardcoded paths configurable with sensible defaults
- [ ] Add startup validation that fails fast with clear error messages

**Affected Files**:
- `src/app/config.py`
- `docker-compose.yml`
- `infra/helm/ml-model-chart/values.yaml`
- `docs/CONFIGURATION.md` (new)
- `README.md`

---

### Issue 5: [TESTING] Comprehensive Testing Gaps

**Priority**: High
**Labels**: testing, quality, priority:high

**Description**:
Significant gaps in test coverage across unit, integration, performance, and security testing:

**1. Limited Unit Test Coverage**
- Only 1538 lines of tests for entire codebase
- No tests for drift monitoring service (`src/drift_monitoring/service.py`)
- No tests for correlation middleware
- No tests for predict endpoint validation logic

**2. No Integration Tests**
- Missing end-to-end API tests
- No MLflow integration tests
- No Loki integration tests for drift monitoring
- No Kubernetes deployment smoke tests

**3. No Load/Performance Tests**
- No performance benchmarks or load testing
- Unknown system limits and capacity

**4. No Security Tests**
- No authentication/authorization tests for admin endpoint
- No input validation/fuzzing tests
- No injection tests for Loki queries

**5. Inadequate Error Scenario Testing**
- Tests primarily cover happy path
- Missing: MLflow unavailable, model load failures, malformed inputs, circuit breaker states

**Acceptance Criteria**:
- [ ] Increase unit test coverage to >80%
- [ ] Add integration test suite for all external dependencies
- [ ] Implement load tests using locust or k6 for /predict endpoint
- [ ] Add security test suite (OWASP Top 10 scenarios)
- [ ] Add error scenario tests for all critical paths
- [ ] Configure pytest-cov to fail CI if coverage drops below threshold
- [ ] Add pytest-xdist for parallel test execution

**Affected Files**:
- `tests/` (expand significantly)
- `.github/workflows/ci-lint-test.yml`
- `pyproject.toml`
- `tests/integration/` (new)
- `tests/load/` (new)
- `tests/security/` (new)

---

### Issue 6: [CI/CD] CI/CD Pipeline Improvements

**Priority**: High
**Labels**: cicd, devops, priority:high

**Description**:
Several issues in the CI/CD pipeline that affect security, reliability, and deployment safety:

**1. Secrets in Workflow Files**
- **Location**: `.github/workflows/deploy-canary-and-promote.yml:21-22`
- Model registry name and stage hardcoded instead of using repository variables

**2. No Workflow Approval Gates**
- Deployment workflows have no manual approval step
- Risk of accidental production deployments

**3. No Artifact Attestation**
- Built containers not signed or attested
- Supply chain vulnerability

**4. Missing Rollback Automation**
- Rollback mentioned in runbook but not automated
- No one-click rollback capability

**5. No Security Scanning**
- No `safety check`, `pip-audit`, or container scanning
- Vulnerable dependencies may be deployed

**6. No Parallel Test Execution**
- Tests run serially, slowing down CI

**Acceptance Criteria**:
- [ ] Move hardcoded values to GitHub Actions variables/secrets
- [ ] Add environment protection rules with required reviewers for production
- [ ] Implement Cosign/SLSA attestation for container images
- [ ] Create automated rollback workflow
- [ ] Add security scanning steps (safety, pip-audit, trivy)
- [ ] Enable pytest-xdist for parallel test execution
- [ ] Add explicit error handling for all deployment failure cases

**Affected Files**:
- `.github/workflows/deploy-canary-and-promote.yml`
- `.github/workflows/ci-lint-test.yml`
- `.github/workflows/train-model.yml`
- `.github/workflows/rollback.yml` (new)
- `ci/runbooks/deploy-runbook.md`

---

### Issue 7: [MONITORING] Monitoring and Observability Gaps

**Priority**: High
**Labels**: monitoring, observability, priority:high

**Description**:
Critical gaps in monitoring and observability that hinder debugging and incident response:

**1. No Distributed Tracing for Model Manager**
- **Location**: `src/models/manager.py`
- Model download/load operations not traced
- Difficult to debug slow model loads

**2. Missing Custom Metrics**
- No metrics for: model cache hit/miss rate, model load duration, drift detection frequency, admin endpoint usage

**3. No Alerting Rules Defined**
- **Location**: `infra/monitoring/ml-recording-rules.yaml`
- Only recording rules exist, no PrometheusRule for alerts

**4. Insufficient Health Check**
- **Location**: `src/app/api/health.py`
- Health check doesn't validate model functionality
- Should include shallow inference test

**5. No SLI/SLO Definitions**
- No formal SLIs/SLOs defined for the service
- Cannot measure reliability objectively

**Acceptance Criteria**:
- [ ] Add OpenTelemetry spans to all model operations
- [ ] Implement custom metrics for cache operations, model loading, drift detection
- [ ] Create PrometheusRule with alerts for: high error rate, drift detection, model load failures, circuit breaker open
- [ ] Enhance health check to perform shallow inference test
- [ ] Define and document SLIs/SLOs (latency p99 < 100ms, availability > 99.9%)
- [ ] Create Grafana dashboards for key metrics
- [ ] Document alerting thresholds and runbooks

**Affected Files**:
- `src/models/manager.py`
- `src/app/api/health.py`
- `src/utils/tracing.py`
- `infra/monitoring/ml-recording-rules.yaml`
- `infra/monitoring/alerts.yaml` (new)
- `infra/monitoring/dashboards/` (new)
- `docs/SLI_SLO.md` (new)

---

### Issue 8: [CODE-QUALITY] Code Quality Improvements

**Priority**: High
**Labels**: code-quality, technical-debt, priority:high

**Description**:
Multiple code quality issues that affect maintainability and reliability:

**1. Broad Exception Catching**
- **Locations**: `src/app/main.py:186, 223, 257`, `src/models/infer.py:82, 93, 100`
- Using `except Exception` catches all exceptions including `KeyboardInterrupt`, `SystemExit`

**2. Inconsistent Error Handling**
- **Location**: `src/models/manager.py:470`
- `_load_metrics_from_directory` returns `(None, None)` on error but logs as warning
- Silent failures make debugging difficult

**3. Magic Numbers and Hardcoded Values**
- **Location**: `src/app/api/predict.py:24-25`
- Feature dimension hardcoded to 10 (should be 4)

**4. Missing Type Hints**
- **Location**: `src/resilient_mlflow.py:197`
- Decorator return type not properly annotated

**5. Redundant Code in Helm Templates**
- **Location**: `infra/helm/ml-model-chart/templates/deployment-{stable,canary}.yaml`
- 90% code duplication between stable and canary deployments

**6. Logging Inconsistencies**
- Mix of structured and unstructured logging
- Some log lines not JSON-formatted

**Acceptance Criteria**:
- [ ] Replace `except Exception` with specific exception types
- [ ] Define consistent error handling strategy (fail-fast vs graceful degradation)
- [ ] Remove magic numbers, use constants or derive from model metadata
- [ ] Add complete type hints to all modules
- [ ] Use Helm named templates to DRY deployment templates
- [ ] Standardize on structured JSON logging throughout
- [ ] Add mypy strict mode to CI pipeline
- [ ] Add pylint/ruff checks for code quality

**Affected Files**:
- `src/app/main.py`
- `src/models/infer.py`
- `src/models/manager.py`
- `src/resilient_mlflow.py`
- `src/app/api/predict.py`
- `infra/helm/ml-model-chart/templates/`
- `src/drift_monitoring/monitor.py`
- `pyproject.toml`

---

## 🟢 MEDIUM PRIORITY

### Issue 9: [PERFORMANCE] Scalability Improvements

**Priority**: Medium
**Labels**: performance, scalability, priority:medium

**Description**:
Several scalability concerns that may impact performance under load:

**1. Single-Threaded Model Loading**
- **Location**: `src/models/manager.py:138`
- `asyncio.Lock()` serializes all model operations
- Could block requests under high auto-refresh frequency

**2. No Caching for Predictions**
- Every prediction hits model, no memoization for identical inputs
- Inefficient for repeated queries

**3. Unbounded Log Collection in Drift Monitor**
- **Location**: `src/drift_monitoring/monitor.py:416`
- Hardcoded limit of 5000, could overwhelm memory

**4. No Database for Prediction History**
- Predictions only logged, not stored queryable
- Cannot perform historical analysis or A/B testing

**5. Model Cache Not Shared Across Pods**
- **Location**: `infra/helm/ml-model-chart/templates/deployment-stable.yaml:119`
- Uses `emptyDir: {}`, each pod downloads model independently

**Acceptance Criteria**:
- [ ] Implement read-write lock pattern for model manager (allow concurrent reads)
- [ ] Add LRU cache for predictions with configurable TTL
- [ ] Implement streaming/pagination for large Loki log queries
- [ ] Add optional prediction history persistence (TimescaleDB or InfluxDB)
- [ ] Use shared PVC or init container for model caching
- [ ] Add load tests to validate scalability improvements

**Affected Files**:
- `src/models/manager.py`
- `src/app/api/predict.py`
- `src/drift_monitoring/monitor.py`
- `infra/helm/ml-model-chart/templates/deployment-stable.yaml`
- `infra/helm/ml-model-chart/templates/pvc.yaml` (new)
- `tests/load/` (new)

---

### Issue 10: [SECURITY] Container and Deployment Security Hardening

**Priority**: Medium
**Labels**: security, kubernetes, docker, priority:medium

**Description**:
Container and Kubernetes deployment security improvements:

**1. Container Runs as Root**
- **Location**: `docker/Dockerfile`
- No `USER` directive, runs as root (UID 0)

**2. No Resource Limits in docker-compose**
- **Location**: `docker-compose.yml`
- No `mem_limit` or `cpus` constraints

**3. Missing Liveness Probe**
- **Location**: `infra/helm/ml-model-chart/templates/deployment-stable.yaml`
- Only `readinessProbe` defined, no `livenessProbe`

**4. Inadequate Pod Security Context**
- No `securityContext` defined
- Missing: `runAsNonRoot`, `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem`

**5. No Network Policy for Egress**
- **Location**: `infra/helm/ml-model-chart/templates/networkpolicy.yaml`
- Egress to MLflow, Loki, Jaeger disabled by default

**6. Unpinned Docker Base Image**
- **Location**: `docker/Dockerfile:4`
- `FROM python:3.11-slim` not pinned to digest

**Acceptance Criteria**:
- [ ] Add non-root user to Dockerfile (UID 1000)
- [ ] Add resource limits to docker-compose.yml
- [ ] Add liveness probe with appropriate timeout
- [ ] Add restrictive pod security context to Helm templates
- [ ] Enable egress network policies with proper selectors
- [ ] Pin Docker base image to specific digest
- [ ] Add init container for model pre-fetch to reduce startup time

**Affected Files**:
- `docker/Dockerfile`
- `docker-compose.yml`
- `infra/helm/ml-model-chart/templates/deployment-*.yaml`
- `infra/helm/ml-model-chart/templates/networkpolicy.yaml`

---

### Issue 11: [DEPS] Dependencies and Version Management

**Priority**: Medium
**Labels**: dependencies, security, priority:medium

**Description**:
Issues with dependency management and versioning:

**1. Loose Version Constraints**
- **Location**: `requirements.txt`
- Some packages are outdated: `fastapi==0.100.0` (current 0.109+), `mlflow==2.7.0` (current 2.10+)

**2. Duplicate Dependency Management**
- Both `requirements.txt` and `pyproject.toml` exist
- Need single source of truth

**3. Missing Dependency Licenses Check**
- No verification of dependency licenses
- May include GPL/AGPL dependencies

**4. Python JSON Logger Fallback Issue**
- **Location**: `src/utils/logging.py:14-18`
- Falls back to text logging if `python-json-logger` import fails
- Breaks Loki JSON parsing expectations

**Acceptance Criteria**:
- [ ] Update all dependencies to latest stable versions
- [ ] Configure dependabot for automated updates
- [ ] Choose single dependency management approach (prefer pyproject.toml)
- [ ] Add `pip-licenses` check to CI
- [ ] Make `python-json-logger` a required dependency (no fallback)
- [ ] Add license compliance documentation

**Affected Files**:
- `requirements.txt`
- `pyproject.toml`
- `src/utils/logging.py`
- `.github/dependabot.yml`
- `.github/workflows/ci-lint-test.yml`
- `docs/LICENSES.md` (new)

---

### Issue 12: [DOCS] Documentation and MLOps Improvements

**Priority**: Medium
**Labels**: documentation, mlops, priority:medium

**Description**:
Documentation gaps and MLOps process improvements:

**1. No Architecture Decision Records (ADRs)**
- No documentation of architectural choices
- Examples: Why Evidently? Why runtime model loading?

**2. Missing Monitoring Dashboard Documentation**
- No documentation of Grafana dashboards, alert definitions

**3. No Model Versioning Strategy**
- No documented strategy for model version lifecycle
- Questions: When to archive? How many to keep? Rollback procedure?

**4. Missing Model Explainability**
- No SHAP values or feature importance exposed
- Difficult to debug predictions

**5. No A/B Testing Framework**
- Cannot easily compare model versions in production

**6. No Feature Store Integration**
- Features computed ad-hoc, no reusability

**7. No Schema Versioning**
- No versioning for data schemas or model contracts

**8. Missing Data Validation on Input**
- **Location**: `src/app/api/predict.py:97-118`
- Only validates shape, not value ranges (NaN, inf, negatives)

**Acceptance Criteria**:
- [ ] Create ADRs for major architectural decisions
- [ ] Document all Grafana dashboards and alerts
- [ ] Define and document model lifecycle policy
- [ ] Add optional explainability endpoint (SHAP values)
- [ ] Leverage canary infrastructure for A/B testing
- [ ] Add schema versioning to model artifacts and data contracts
- [ ] Add value range validation based on training data distribution
- [ ] Create comprehensive runbook with incident response procedures
- [ ] Evaluate feature store for future implementation (Feast, Tecton)

**Affected Files**:
- `docs/adr/` (new)
- `docs/MONITORING.md` (new)
- `docs/MODEL_LIFECYCLE.md` (new)
- `src/app/api/explain.py` (new)
- `src/app/api/predict.py`
- `ci/runbooks/incident-response.md` (new)
- `docs/ROADMAP.md` (new)

---

## Summary Statistics

- **Total Issues**: 12 formulated issues
- **Critical Priority**: 3 issues
- **High Priority**: 5 issues
- **Medium Priority**: 4 issues
- **Total Findings**: 68 specific issues covered

## Recommended Implementation Order

1. **Sprint 1 (Immediate - 1-2 weeks)**:
   - Issue #2: Feature dimension mismatch (critical bug)
   - Issue #1: Security vulnerabilities (partial: tokens, input validation, rate limiting)
   - Issue #10: Container security (non-root user)

2. **Sprint 2 (Short-term - 3-4 weeks)**:
   - Issue #1: Security vulnerabilities (complete: TLS, Loki injection)
   - Issue #3: Secret management
   - Issue #4: Configuration management
   - Issue #6: CI/CD improvements (partial: security scanning, approval gates)

3. **Sprint 3 (1-2 months)**:
   - Issue #5: Testing gaps
   - Issue #7: Monitoring and observability
   - Issue #8: Code quality improvements

4. **Sprint 4 (2-3 months)**:
   - Issue #9: Scalability improvements
   - Issue #6: CI/CD improvements (complete: attestation, rollback)
   - Issue #11: Dependencies management

5. **Sprint 5 (3-6 months)**:
   - Issue #12: Documentation and MLOps maturity
   - Feature store evaluation
   - A/B testing framework

---

## Notes for Implementation

1. **Testing**: Each issue should include comprehensive tests before closing
2. **Documentation**: Update relevant documentation as part of each issue
3. **Security**: All security issues should be treated with high priority
4. **Backward Compatibility**: Ensure changes don't break existing deployments
5. **Monitoring**: Add metrics and alerts for new features

---

Generated: 2025-11-07
Review Type: Comprehensive Architecture Review
Codebase: ML CI/CD Pipeline
