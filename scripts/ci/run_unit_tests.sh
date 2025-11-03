#!/usr/bin/env bash
set -euo pipefail
# Run unit tests locally in a POSIX shell. For Windows use PowerShell activation.
python -m pip install --upgrade pip
pip install poetry
poetry config virtualenvs.create false || true
poetry install --no-interaction --no-ansi
poetry run pytest -q
