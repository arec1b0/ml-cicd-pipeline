<#
.SYNOPSIS
  Windows 11 initialization script for the repository.

.DESCRIPTION
  - Validates presence of Python.
  - Creates a virtual environment (.venv) if Poetry not used.
  - Installs Poetry if absent and runs `poetry install`.
  - Falls back to pip install -r requirements.txt when Poetry missing.
  - Outputs clear next-step instructions for Windows PowerShell.

.NOTES
  This script is intended for developer workstations on Windows 11.
  Keep operations idempotent and explicit. Comments follow high-technology documentation style.
#>

param()

# Fail on errors
$ErrorActionPreference = "Stop"

Write-Host "Starting repo bootstrap (Windows 11)."

# Check Python
try {
    $pyVersionOutput = & python --version 2>&1
} catch {
    Write-Error "Python not found in PATH. Install Python 3.11 and retry."
    exit 1
}

if ($pyVersionOutput -notmatch "3\.11") {
    Write-Warning "Detected Python version: $pyVersionOutput. Recommended: Python 3.11.x."
} else {
    Write-Host "Python 3.11 detected: $pyVersionOutput"
}

# Try Poetry first
$poetry = Get-Command poetry -ErrorAction SilentlyContinue

if (-not $poetry) {
    Write-Host "Poetry not found. Attempting to install Poetry (user-level)."
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "https://install.python-poetry.org" -OutFile "$env:TEMP\install-poetry.py"
        & python "$env:TEMP\install-poetry.py"
        Remove-Item "$env:TEMP\install-poetry.py" -Force
        Write-Host "Poetry installation attempted. You may need to restart shell or add Poetry to PATH."
    } catch {
        Write-Warning "Automatic Poetry install failed. You can install Poetry manually: https://python-poetry.org/docs/#installation"
    }
    # refresh command lookup
    $poetry = Get-Command poetry -ErrorAction SilentlyContinue
}

if ($poetry) {
    Write-Host "Poetry found. Running 'poetry install'."
    & poetry install
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "poetry install failed. Check output."
        exit 1
    }
    Write-Host "Poetry dependencies installed. To activate Poetry shell use 'poetry shell'."
    exit 0
}

# Fallback: create .venv and pip install
Write-Host "Poetry unavailable. Creating .venv and installing requirements.txt as fallback."

python -m venv .venv
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment creation failed."
    exit 1
}

Write-Host "Activating virtual environment."
# Note: user must run the activation command in their interactive shell:
Write-Host "Run in your interactive PowerShell: `& .\.venv\Scripts\Activate.ps1`"

# Install packages using pip
& .\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed. Inspect output."
    exit 1
}

Write-Host "Fallback environment prepared. Activate with: `& .\.venv\Scripts\Activate.ps1`"
