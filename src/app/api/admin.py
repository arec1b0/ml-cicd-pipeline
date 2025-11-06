"""
Administrative endpoints for runtime operations (e.g., model reload).
"""

from __future__ import annotations
import logging
import secrets
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class ReloadResponse(BaseModel):
    """Response model for the /admin/reload endpoint.

    Attributes:
        status: The status of the reload operation (e.g., "reloaded", "noop").
        detail: A message describing the outcome of the reload.
        version: The version of the newly loaded model, if applicable.
        stage: The stage of the newly loaded model, if applicable.
    """
    status: str
    detail: str
    version: str | None = None
    stage: str | None = None


@router.post("/reload", response_model=ReloadResponse, status_code=status.HTTP_200_OK)
async def reload_model(request: Request) -> ReloadResponse:
    """Triggers a forced reload of the machine learning model.

    This endpoint provides an administrative function to force the service to
    reload the model from its source. It is protected by an admin token.

    Args:
        request: The incoming FastAPI request object.

    Raises:
        HTTPException: If the admin token is missing or invalid (403),
                       if the model manager is not available (503), or if an
                       error occurs during the reload (500).

    Returns:
        A ReloadResponse object indicating the outcome of the operation.
    """
    app = request.app
    header_name = getattr(app.state, "admin_token_header", "X-Admin-Token")
    expected_token = getattr(app.state, "admin_api_token", None)
    provided_token = request.headers.get(header_name)

    if expected_token:
        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    manager = getattr(app.state, "model_manager", None)
    apply_fn = getattr(app.state, "apply_model_state", None)
    if manager is None or apply_fn is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model manager unavailable")

    try:
        state = await manager.reload(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual model reload failed", extra={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Reload failed: {exc}") from exc

    if state is None:
        return ReloadResponse(status="noop", detail="Model descriptor unchanged")

    apply_fn(state)

    logger.info(
        "Manual model reload applied",
        extra={
            "model_uri": state.descriptor.model_uri,
            "version": state.descriptor.version,
            "stage": state.descriptor.stage,
        },
    )

    return ReloadResponse(
        status="reloaded",
        detail="Model reloaded successfully",
        version=state.descriptor.version,
        stage=state.descriptor.stage,
    )
