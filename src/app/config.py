"""
Configuration loader for the inference service.
Reads configuration from environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path

def get_env(name: str, default: str | None = None) -> str | None:
    """
    Read environment variable with fallback.
    """
    val = os.getenv(name, default)
    return val

MODEL_PATH = Path(get_env("MODEL_PATH", "/app/model_registry/model.pkl"))
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
