"""
Model explainability endpoint.

Provides SHAP-based explanations for model predictions, enabling debugging and
interpretability of individual predictions. This helps data scientists and engineers
understand why a model made a particular prediction.

Comments and docstrings are written in English per repo standard.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.app.api.predict import PredictRequest
from src.utils.drift import emit_prediction_log
from src.utils.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplainResponse(BaseModel):
    """Response model for the /explain endpoint.
    
    Attributes:
        prediction: The model's prediction for the input features.
        shap_values: SHAP values for each feature (relative importance).
        feature_values: The input feature values used for prediction.
        base_value: The model's expected value (intercept) when no features are considered.
        explanation_type: Type of explanation method used (e.g., "tree_shap", "kernel_shap").
    """
    prediction: int
    shap_values: List[float]
    feature_values: List[float]
    base_value: Optional[float] = None
    explanation_type: str


@router.post("/", response_model=ExplainResponse)
async def explain(request: Request, payload: PredictRequest, background_tasks: BackgroundTasks) -> ExplainResponse:
    """Provides SHAP-based explanations for model predictions.
    
    This endpoint accepts a single feature vector and returns SHAP values that explain
    the model's prediction. SHAP (SHapley Additive exPlanations) values provide a unified
    measure of each feature's contribution to the prediction.
    
    Args:
        request: The incoming FastAPI request object.
        payload: The request body containing a single feature vector for explanation.
        background_tasks: FastAPI's background task runner.
    
    Raises:
        HTTPException: If the model is not loaded (503), the input is invalid (400),
                      or explanation generation fails (500).
    
    Returns:
        ExplainResponse: An object containing the prediction and SHAP-based explanations.
    """
    app = request.app
    correlation_id = getattr(request.state, "correlation_id", None)
    
    # Check readiness flag and wrapper
    model_wrapper = getattr(app.state, "ml_wrapper", None)
    is_ready = bool(getattr(app.state, "is_ready", False))
    if not is_ready or model_wrapper is None:
        logger.error("Model not loaded", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # For explainability, we typically want single samples
    # But we accept single sample in batch format for consistency with /predict endpoint
    if len(payload.features) != 1:
        logger.warning(
            "Explain endpoint expects single sample",
            extra={
                "correlation_id": correlation_id,
                "sample_count": len(payload.features)
            }
        )
        raise HTTPException(
            status_code=400,
            detail="Explain endpoint expects exactly one feature vector. For batch explanation, call this endpoint multiple times."
        )
    
    features = payload.features[0]
    
    logger.info(
        "Processing explanation request",
        extra={
            "correlation_id": correlation_id,
            "feature_dim": len(features)
        }
    )
    
    # Get model metadata for span attributes
    model_metadata = getattr(app.state, "model_metadata", None)
    model_path = model_metadata.get("model_path") if model_metadata else None
    
    try:
        # Create custom span for explanation generation
        with tracer.start_as_current_span("model_explanation") as span:
            span.set_attribute("ml.model.path", model_path or "unknown")
            span.set_attribute("ml.explanation.type", "shap")
            if correlation_id:
                span.set_attribute("correlation.id", correlation_id)
            
            # Generate prediction
            prediction = model_wrapper.predict([features])[0]
            
            # Generate SHAP explanation
            shap_values, base_value, explanation_type = _generate_shap_explanation(
                model_wrapper,
                np.array([features], dtype=np.float32),
                features
            )
            
            span.set_attribute("ml.explanation.generated", True)
            
            logger.info(
                "Explanation generated successfully",
                extra={
                    "correlation_id": correlation_id,
                    "explanation_type": explanation_type,
                    "prediction": prediction
                }
            )
    
    except Exception as exc:
        logger.exception(
            "Explanation generation failed",
            extra={"correlation_id": correlation_id, "error": str(exc)}
        )
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")
    
    # Convert to appropriate types
    if isinstance(prediction, float) and prediction.is_integer():
        prediction = int(prediction)
    
    response = ExplainResponse(
        prediction=prediction,
        shap_values=shap_values,
        feature_values=features,
        base_value=base_value,
        explanation_type=explanation_type
    )
    
    # Log explanation metadata
    metadata = {
        "path": str(request.url.path),
        "client_host": getattr(request.client, "host", None),
        "model": model_metadata,
        "headers": {
            "user_agent": request.headers.get("user-agent"),
            "x_request_id": request.headers.get("x-request-id"),
        },
    }
    background_tasks.add_task(
        emit_prediction_log,
        features=[features],
        predictions=[prediction],
        metadata=metadata,
    )
    
    return response


def _generate_shap_explanation(
    model_wrapper,
    features: np.ndarray,
    original_features: List[float]
) -> tuple[List[float], Optional[float], str]:
    """Generate SHAP-based explanations for a model prediction.
    
    This function attempts to generate SHAP values using different methods depending
    on the model type. It first tries TreeExplainer (for tree-based models like
    RandomForest), then falls back to simpler feature importance methods.
    
    Args:
        model_wrapper: The ModelWrapper instance containing the model.
        features: Feature array for SHAP calculation (numpy array).
        original_features: Original feature values (list of floats).
    
    Returns:
        tuple: (shap_values, base_value, explanation_type)
            - shap_values: List of SHAP values for each feature
            - base_value: The model's expected value (intercept)
            - explanation_type: Description of which method was used
    
    Raises:
        RuntimeError: If explanation generation fails with all methods.
    """
    logger.debug("Generating SHAP explanation", extra={})
    
    # Try TreeExplainer for tree-based models (RandomForest, XGBoost, LightGBM)
    try:
        import shap
        
        # Check if model is a tree-based model
        if _is_tree_model(model_wrapper._model):
            explainer = shap.TreeExplainer(model_wrapper._model)
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case: shap_values is list of arrays
                # For binary classification, take the positive class
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Extract values for single sample
            if len(shap_vals.shape) > 1:
                shap_vals_list = shap_vals[0].tolist()
            else:
                shap_vals_list = shap_vals.tolist()
            
            base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None
            
            logger.info("SHAP TreeExplainer succeeded", extra={})
            return shap_vals_list, base_value, "tree_shap"
    
    except ImportError:
        logger.warning("SHAP library not available, using fallback method", extra={})
    except Exception as e:
        logger.warning("TreeExplainer failed, trying fallback method", extra={"error": str(e), "error_type": type(e).__name__})
    
    # Fallback: Use simple feature importance from the model if available
    try:
        if hasattr(model_wrapper._model, 'feature_importances_'):
            # Normalize feature importances to sum to 1
            importances = model_wrapper._model.feature_importances_.astype(float)
            importances = importances / (importances.sum() + 1e-10)
            
            # Scale by feature values to approximate SHAP-like values
            shap_vals_list = (importances * np.abs(features[0])).tolist()
            
            logger.info("Using feature_importances_ as fallback explanation", extra={})
            return shap_vals_list, None, "feature_importance"
    except Exception as e:
        logger.warning("Feature importance fallback failed", extra={"error": str(e), "error_type": type(e).__name__})
    
    # Final fallback: Return zero SHAP values
    logger.warning("All explanation methods failed, returning zero values", extra={})
    shap_vals_list = [0.0] * len(original_features)
    return shap_vals_list, None, "fallback_zero"


def _is_tree_model(model: Any) -> bool:
    """Check if a model is a tree-based model that SHAP TreeExplainer can handle.
    
    Args:
        model: The model object to check.
    
    Returns:
        bool: True if the model is tree-based, False otherwise.
    """
    model_class_name = type(model).__name__
    tree_model_names = {
        'RandomForestClassifier', 'RandomForestRegressor',
        'ExtraTreesClassifier', 'ExtraTreesRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'XGBClassifier', 'XGBRegressor', 'XGBRFClassifier',
        'LGBMClassifier', 'LGBMRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
    }
    return model_class_name in tree_model_names

