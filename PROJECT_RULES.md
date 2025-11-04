# Project Rules and Standards

This document defines the coding standards, practices, and conventions for the ML CI/CD Pipeline project.

## Table of Contents

1. [Python Code Standards](#python-code-standards)
2. [Type Checking](#type-checking)
3. [Testing Requirements](#testing-requirements)
4. [Git Workflow](#git-workflow)
5. [CI/CD Requirements](#cicd-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Security Practices](#security-practices)
8. [ML-Specific Rules](#ml-specific-rules)
9. [Cross-Platform Considerations](#cross-platform-considerations)
10. [Dependency Management](#dependency-management)

---

## Python Code Standards

### Language Version
- **Python 3.11+** is required
- Use modern Python features (type hints, dataclasses, f-strings)
- Prefer `from __future__ import annotations` for forward references

### Code Formatting
- **Ruff** is the primary linter and formatter
- Run `ruff check src tests` before committing
- Follow PEP 8 conventions with Ruff's default settings
- Maximum line length: 100 characters (Ruff default)

### Code Style
- Use type hints for all function signatures and class attributes
- Prefer explicit imports over wildcard imports (`from module import *`)
- Use absolute imports: `from src.app.config import MODEL_PATH`
- Document complex logic with inline comments
- Use descriptive variable and function names
- Functions should have docstrings if not self-explanatory

### Naming Conventions
- **Modules**: `snake_case` (e.g., `trainer.py`, `infer.py`)
- **Classes**: `PascalCase` (e.g., `PrometheusMiddleware`)
- **Functions/Methods**: `snake_case` (e.g., `load_model`, `create_app`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_PATH`, `LOG_LEVEL`)
- **Private**: Prefix with single underscore `_` for internal use

---

## Type Checking

### MyPy Configuration
- **MyPy** must pass with no errors: `mypy src`
- All public APIs must have complete type annotations
- Use `typing` module for complex types (Optional, Union, Dict, List, etc.)
- Avoid `Any` unless absolutely necessary
- Use `Protocol` for structural typing when needed

### Type Annotation Examples
```python
from typing import Optional, Dict, List
from pathlib import Path

def load_model(model_path: Path) -> Optional[sklearn.base.BaseEstimator]:
    """Load a model from the given path."""
    pass

def predict_batch(features: List[Dict[str, float]]) -> List[float]:
    """Make predictions for a batch of features."""
    pass
```

---

## Testing Requirements

### Test Framework
- **pytest** is the testing framework
- All tests must pass: `pytest -q`
- Tests should be fast and isolated
- Use fixtures for common test setup

### Test Organization
- Unit tests: `tests/unit/` (test individual functions/classes)
- Integration tests: `tests/integration/` (test component interactions)
- Data pipeline tests: `tests/test_data_pipeline.py` (validate data schemas)

### Test Coverage
- Aim for >80% code coverage
- Critical paths (model loading, prediction) must have tests
- Data validators must have tests
- API endpoints must have tests

### Test Naming
- Test files: `test_*.py`
- Test functions: `test_<functionality>_<scenario>`
- Example: `test_load_model_with_valid_path()`, `test_predict_raises_on_invalid_input()`

---

## Git Workflow

### Branch Strategy
- `main` branch is protected and requires PR approval
- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- Hotfixes: `hotfix/<description>`

### Commit Messages
- Use conventional commits format:
  - `feat: add drift monitoring service`
  - `fix: correct model loading path handling`
  - `docs: update API documentation`
  - `test: add unit tests for data validators`
  - `refactor: simplify telemetry middleware`
  - `ci: update GitHub Actions workflow`

### Pre-Commit Checks
- Code must pass Ruff linting
- Code must pass MyPy type checking
- All tests must pass
- No secrets or credentials in code

---

## CI/CD Requirements

### GitHub Actions Workflows
All code changes must pass the following CI checks:

1. **Lint & Type Check** (`.github/workflows/ci-lint-test.yml`)
   - Ruff linting on `src` and `tests`
   - MyPy type checking on `src`
   - Pytest execution
   - Runs on Ubuntu and Windows

2. **Data Validation** (`.github/workflows/data-validation.yml`)
   - Executes data validators when data assets change
   - Catches schema drifts early

3. **Model Training** (`.github/workflows/model-training.yml`)
   - Automates model retraining against MLflow
   - Registers new model versions

4. **Deploy & Promote** (`.github/workflows/deploy-canary-and-promote.yml`)
   - Builds and pushes Docker images
   - Deploys Helm canary
   - Runs smoke tests
   - Evaluates model metrics before promotion

### CI/CD Rules
- All workflows must pass before merging to `main`
- Model deployments require:
  - Accuracy threshold: `model.metrics.accuracy >= 0.70`
  - Health check: `/health/` returns `ready=true`
  - Smoke test: `/predict/` returns valid prediction
  - Error rate: HTTP 5xx < 1%
  - Latency: 95th percentile < 500ms (optional)

---

## Documentation Standards

### Code Documentation
- **Docstrings**: Use Google-style docstrings for classes and public functions
- **Inline Comments**: Explain "why" not "what" for complex logic
- **Type Hints**: Serve as inline documentation

### Docstring Example
```python
def load_model(model_path: Path) -> Optional[sklearn.base.BaseEstimator]:
    """
    Load a scikit-learn model from the specified path.
    
    Args:
        model_path: Path to the serialized model file (.pkl or .joblib)
        
    Returns:
        Loaded model instance, or None if loading fails
        
    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If model file is corrupted
    """
    pass
```

### API Documentation
- FastAPI endpoints are auto-documented via OpenAPI/Swagger
- Update `API_DOCUMENTATION.md` when adding/modifying endpoints
- Include request/response examples

### Project Documentation
- Update `README.md` for major changes
- Update `docs/ARCHITECTURE.md` for architectural changes
- Update `docs/SET-UP.md` for setup procedure changes
- Keep runbooks in `ci/runbooks/` up to date

---

## Security Practices

### Secrets Management
- **Never** commit secrets, API keys, or credentials
- Use environment variables for configuration
- `.env` files are gitignored
- Use GitHub Secrets for CI/CD credentials

### Code Security
- Validate and sanitize all user inputs
- Use parameterized queries/inputs for model inference
- Keep dependencies up to date (`poetry update`)
- Review dependency security advisories

### Model Security
- Validate input feature schemas before prediction
- Set reasonable input limits (batch size, feature count)
- Log predictions for audit trails (no PII)
- Monitor for adversarial inputs

---

## ML-Specific Rules

### Model Management
- **MLflow** is the model registry
- All models must be registered with versioning
- Model training must log metrics, parameters, and artifacts
- Reference datasets must be persisted for drift detection

### Model Versioning
- Use semantic versioning: `major.minor.patch`
- Breaking changes increment major version
- Model artifacts should include:
  - Serialized model file
  - Training metadata (metrics, hyperparameters)
  - Feature schema
  - Reference dataset snapshot

### Data Validation
- Implement validators in `src/data/validators/`
- Validators must be executable via CLI
- Validate schema, types, ranges, and distributions
- Fail fast on validation errors

### Monitoring & Observability
- All predictions must emit Prometheus metrics
- Use structured JSON logging for prediction logs
- Implement drift monitoring with Evidently
- Alert on data drift and high prediction PSI

### Model Performance
- Models must meet accuracy threshold (≥0.70) for production
- Monitor prediction latency (P95 < 500ms)
- Track error rates (< 1% HTTP 5xx)
- Implement canary deployments for gradual rollout

---

## Cross-Platform Considerations

### Platform Support
- **Windows 11+** (PowerShell 7+)
- **Linux** (Ubuntu 20.04+)
- **macOS** (optional, via Linux compatibility)

### Scripts
- Use cross-platform Python code when possible
- Platform-specific scripts in `scripts/windows/` and `scripts/linux/`
- Prefer Python scripts over shell scripts for portability
- Use `pathlib.Path` instead of `os.path`

### File Paths
- Use `pathlib.Path` for all file operations
- Avoid hardcoded path separators (`/` or `\`)
- Test on both Windows and Linux in CI

### Environment Variables
- Use environment variables for configuration
- Document required env vars in `docs/SET-UP.md`
- Provide `.env.example` template (if needed)

---

## Dependency Management

### Poetry (Preferred)
- **Poetry** is the preferred dependency manager
- Update `pyproject.toml` for new dependencies
- Run `poetry lock` after adding dependencies
- Commit `poetry.lock` for reproducible builds

### Requirements.txt (Fallback)
- `requirements.txt` is maintained for compatibility
- Keep synchronized with `pyproject.toml`
- Used when Poetry is not available

### Dependency Rules
- Pin major versions, allow minor/patch updates
- Example: `fastapi = "^0.100.0"` (allows 0.100.x, not 0.101.x)
- Review and update dependencies quarterly
- Remove unused dependencies

### Dependency Categories
- **Production**: Required for runtime
- **Development**: Testing, linting, type checking
- **Optional**: Platform-specific or feature flags

---

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code passes Ruff linting (`ruff check src tests`)
- [ ] Code passes MyPy type checking (`mypy src`)
- [ ] All tests pass (`pytest -q`)
- [ ] New code has appropriate tests
- [ ] Type hints are complete and correct
- [ ] Docstrings added for public APIs
- [ ] No secrets or credentials in code
- [ ] Documentation updated if needed
- [ ] Cross-platform compatibility verified
- [ ] CI/CD workflows will pass
- [ ] Commit messages follow conventional format

---

## Enforcement

- **Pre-commit hooks**: Consider adding pre-commit hooks for automatic checks
- **CI/CD gates**: All checks must pass before merge
- **Code review**: At least one approval required for `main` branch
- **Automated tests**: Failures block deployments

---

## Updates to This Document

This document should be updated when:
- New tools or standards are adopted
- Project requirements change
- Team consensus on new practices is reached
- CI/CD pipeline changes require updated rules

**Last Updated**: Generated from project analysis

