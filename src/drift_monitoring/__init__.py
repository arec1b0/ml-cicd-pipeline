"""
Drift monitoring service package.
Provides FastAPI app factory exposed as ``drift_monitoring.service:create_app``.
"""

from .service import create_app

__all__ = ["create_app"]
