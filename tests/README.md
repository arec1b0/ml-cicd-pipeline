# Testing Documentation

This document describes the comprehensive test suite for the ML CI/CD pipeline project.

## Test Organization

The test suite is organized into the following categories:

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_*.py       # Unit test files
├── integration/        # Integration tests for API and services
│   ├── test_api_endpoints.py
├── load/              # Load and performance tests
│   ├── locustfile.py  # Locust load tests
├── security/          # Security tests (OWASP Top 10)
│   ├── test_security.py
└── README.md          # This file
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
poetry install
```

### Run All Tests

Run the complete test suite with coverage:

```bash
poetry run pytest
```

This will:
- Run all tests in parallel (`-n auto`)
- Generate coverage reports (term, HTML, XML)
- Fail if coverage is below 80%

### Run Specific Test Types

**Unit tests only:**
```bash
poetry run pytest tests/unit/
```

**Integration tests:**
```bash
poetry run pytest tests/integration/ -m integration
```

**Security tests:**
```bash
poetry run pytest tests/security/ -m security
```

**Exclude slow tests:**
```bash
poetry run pytest -m "not slow"
```

### Coverage Reports

**Terminal coverage report:**
```bash
poetry run pytest --cov=src --cov-report=term-missing
```

**HTML coverage report:**
```bash
poetry run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

**XML coverage report (for CI):**
```bash
poetry run pytest --cov=src --cov-report=xml
```

### Load Testing

Run load tests using Locust:

**With Web UI:**
```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser.

**Headless mode:**
```bash
# 100 users, spawn rate 10/sec, run for 60 seconds
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 60s \
    --headless \
    --csv=results/load_test
```

**Stress testing:**
```bash
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 500 \
    --spawn-rate 50 \
    --run-time 120s \
    --headless
```

**Test specific endpoints:**
```bash
# Only test predict endpoint
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --tags predict \
    --users 100 \
    --spawn-rate 10 \
    --run-time 60s
```

## Test Categories

### Unit Tests

Unit tests verify individual components in isolation:

- **test_drift_service.py**: Drift monitoring service
- **test_correlation_middleware.py**: Correlation ID middleware
- **test_telemetry.py**: Prometheus telemetry middleware
- **test_explain.py**: SHAP explanation endpoint
- **test_feature_statistics.py**: Feature statistics and validation
- **test_error_scenarios.py**: Error handling and edge cases
- **test_infer.py**: Model inference
- **test_model_manager.py**: Model management and reloading
- **test_resilient_mlflow.py**: Circuit breaker and retry logic
- **test_admin_reload.py**: Admin authentication and model reload
- **test_health_with_reload.py**: Health endpoint states
- **test_predict_validation.py**: Prediction input validation

### Integration Tests

Integration tests verify end-to-end functionality:

- **test_api_endpoints.py**: Full API integration
  - Health endpoint with model states
  - Predict endpoint with real predictions
  - Explain endpoint with SHAP
  - Admin reload endpoint
  - Middleware integration
  - Concurrent request handling

### Load Tests

Performance and load tests using Locust:

- **MLPredictionUser**: Normal user behavior
  - Single sample predictions (most common)
  - Small batch predictions (5-10 samples)
  - Large batch predictions (50-100 samples)
  - Explanation requests
  - Health checks

- **MLPredictionStressUser**: Stress testing
  - Rapid requests with minimal wait time
  - Tests system limits

- **MLPredictionEdgeCaseUser**: Edge case testing
  - Invalid features
  - Wrong dimensions
  - Special numeric values (NaN, Inf)

### Security Tests

Security tests cover OWASP Top 10 scenarios:

1. **Authentication & Authorization**
   - Admin endpoint authentication
   - Token validation
   - Malformed auth headers

2. **Injection Prevention**
   - LogQL injection (Loki queries)
   - JSON payload injection
   - Header injection (CRLF)

3. **Input Validation**
   - Feature dimension validation
   - Numeric range validation
   - Batch size limits
   - Run ID format validation

4. **Sensitive Data Exposure**
   - Error message sanitization
   - No internal paths in responses

5. **Rate Limiting & DoS**
   - Rapid request handling
   - Large payload handling

6. **SSRF Prevention**
   - Model URI validation

7. **Fuzzing**
   - Random JSON inputs

### Error Scenario Tests

Tests for error handling and resilience:

- **MLflow Unavailable**
  - Circuit breaker behavior
  - Degraded operation
  - Connection errors
  - Timeouts

- **Model Load Failures**
  - Corrupted model files
  - Version incompatibility
  - Missing models

- **Malformed Inputs**
  - Null/missing features
  - Wrong types
  - NaN/Inf values
  - Dimension mismatches

- **Resource Exhaustion**
  - Large batches
  - High-dimensional features
  - Concurrent requests

## Coverage Goals

The test suite aims for >80% code coverage:

- **Target**: 80% overall coverage
- **Enforcement**: CI pipeline fails if coverage drops below 80%
- **Reports**: Generated in HTML, XML, and terminal formats

### Current Coverage

Run this command to see current coverage:

```bash
poetry run pytest --cov=src --cov-report=term-missing
```

## Continuous Integration

The CI pipeline (`.github/workflows/ci-lint-test.yml`) runs:

1. **Linting**: `ruff` on src and tests
2. **Type Checking**: `mypy` on src
3. **Tests**: `pytest` with coverage enforcement
4. **Security Scanning**: `bandit`, `safety`, `pip-audit`

Tests run on:
- **Platforms**: Ubuntu, Windows
- **Python**: 3.11

Coverage reports are uploaded as artifacts (30-day retention).

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for <module_name>.

Tests cover:
- <functionality 1>
- <functionality 2>
"""

import pytest
from unittest.mock import Mock, patch

def test_<functionality>():
    """Test that <specific behavior>."""
    # Arrange

    # Act

    # Assert
    assert result == expected
```

### Integration Test Template

```python
"""
Integration tests for <component>.
"""

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def integration_app():
    """Create fully configured app."""
    # Setup
    yield app
    # Teardown

def test_<scenario>(integration_app):
    """Test end-to-end <scenario>."""
    client = TestClient(integration_app)
    response = client.get("/endpoint")
    assert response.status_code == 200
```

### Best Practices

1. **Isolation**: Use mocks to isolate units under test
2. **Fixtures**: Use pytest fixtures for reusable test setup
3. **Descriptive names**: Test names should describe what they test
4. **AAA pattern**: Arrange, Act, Assert
5. **One assertion**: Focus on one behavior per test (when possible)
6. **Edge cases**: Test boundary conditions and error paths
7. **Documentation**: Add docstrings explaining test purpose

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_full_api_flow():
    pass

@pytest.mark.security
def test_injection_prevention():
    pass
```

## Troubleshooting

### Tests Failing Locally

1. **Install dependencies**: `poetry install`
2. **Update dependencies**: `poetry update`
3. **Clear cache**: `pytest --cache-clear`
4. **Run with verbose**: `pytest -vv`

### Coverage Not Meeting Threshold

1. **Check coverage report**: `pytest --cov-report=html`
2. **Identify uncovered lines**: Open `htmlcov/index.html`
3. **Add tests for uncovered code**
4. **Run coverage again**: `poetry run pytest`

### Load Tests Failing

1. **Start the application**: `uvicorn src.app.main:app --host 0.0.0.0 --port 8000`
2. **Verify health endpoint**: `curl http://localhost:8000/health`
3. **Run load tests**: `locust -f tests/load/locustfile.py --host=http://localhost:8000`

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Locust documentation](https://docs.locust.io/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
