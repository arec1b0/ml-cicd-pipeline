# ml-cicd-pipeline

Senior-level CI/CD scaffold for ML models.  
Windows 11 friendly. Contains GitHub Actions workflows, Kubernetes Helm manifests, MLflow integration, and Windows setup scripts.

## Quick start (Windows PowerShell)

1. Open PowerShell in project root.
2. Run `.\scripts\windows\setup.ps1` to prepare environment.
3. See `.github/workflows/` for CI/CD pipeline definitions.

## Conventions

- Python 3.11
- Poetry preferred. Fallback to requirements.txt available.
- SRP/OCP/LSP/ISP/DIP principles applied to code layout.
