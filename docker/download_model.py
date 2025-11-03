"""
MLflow Model Download Script.
Downloads a model from MLflow Model Registry and saves it to a local path.
Used during Docker image build to fetch the production model.
"""

import mlflow
import click
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-uri",
    required=True,
    help="MLflow model URI (e.g., models:/iris-random-forest/production)"
)
@click.option(
    "--dst-path",
    required=True,
    type=click.Path(),
    help="Destination path to save the downloaded model"
)
@click.option(
    "--onnx",
    is_flag=True,
    default=False,
    help="Download ONNX model instead of sklearn model"
)
def download_model(model_uri: str, dst_path: str, onnx: bool) -> None:
    """
    Download a model from MLflow and save it locally.

    When downloading a registered model, MLflow creates a subdirectory
    structure. For sklearn models, the structure is typically:
    dst_path/model/model.pkl. For ONNX models, it's typically:
    dst_path/model_onnx/model.onnx.

    Args:
        model_uri: MLflow model URI in format
            models:/<model_name>/<stage|version>
        dst_path: Local destination path for the downloaded model
        onnx: If True, download ONNX model; otherwise download sklearn model
    """
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("Using MLflow tracking URI: %s", tracking_uri)
        else:
            logger.warning("MLFLOW_TRACKING_URI not set; using MLflow defaults.")

        logger.info(f"Downloading model from MLflow: {model_uri}")
        dst_path_obj = Path(dst_path)
        dst_path_obj.mkdir(parents=True, exist_ok=True)

        # Download artifacts - MLflow creates subdirectory structure
        # This downloads all artifacts including both sklearn and ONNX models
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(dst_path_obj)
        )

        if onnx:
            # Find the ONNX model file
            # MLflow typically creates: dst_path/model_onnx/model.onnx for ONNX
            model_file = dst_path_obj / "model_onnx" / "model.onnx"
            if not model_file.exists():
                # Fallback: search for model.onnx recursively
                found_models = list(dst_path_obj.rglob("model.onnx"))
                if found_models:
                    model_file = found_models[0]
                    logger.info(f"Found ONNX model at: {model_file}")
                else:
                    # If ONNX not found, try to find sklearn model as fallback
                    logger.warning(
                        f"ONNX model file (model.onnx) not found. "
                        f"Will fall back to sklearn model during inference."
                    )
                    model_file = None
            if model_file:
                logger.info(
                    f"ONNX model successfully downloaded. "
                    f"Model file located at: {model_file}"
                )
        else:
            # Find the sklearn model.pkl file
            # MLflow typically creates: dst_path/model/model.pkl for sklearn
            model_file = dst_path_obj / "model" / "model.pkl"
            if not model_file.exists():
                # Fallback: search for model.pkl recursively
                found_models = list(dst_path_obj.rglob("model.pkl"))
                if found_models:
                    model_file = found_models[0]
                    logger.info(f"Found sklearn model at: {model_file}")
                else:
                    raise FileNotFoundError(
                        f"Model file (model.pkl) not found in downloaded "
                        f"structure at {dst_path}"
                    )
            logger.info(
                f"Model successfully downloaded. "
                f"Model file located at: {model_file}"
            )
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()
