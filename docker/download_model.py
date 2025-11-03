"""
MLflow Model Download Script.
Downloads a model from MLflow Model Registry and saves it to a local path.
Used during Docker image build to fetch the production model.
"""

import mlflow
import click
import logging
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
def download_model(model_uri: str, dst_path: str) -> None:
    """
    Download a model from MLflow and save it locally.

    When downloading a registered model, MLflow creates a subdirectory
    structure. For sklearn models, the structure is typically:
    dst_path/model/model.pkl. This function downloads and finds the
    actual model.pkl file location.

    Args:
        model_uri: MLflow model URI in format
            models:/<model_name>/<stage|version>
        dst_path: Local destination path for the downloaded model
    """
    try:
        logger.info(f"Downloading model from MLflow: {model_uri}")
        dst_path_obj = Path(dst_path)
        dst_path_obj.mkdir(parents=True, exist_ok=True)

        # Download artifacts - MLflow creates subdirectory structure
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(dst_path_obj)
        )

        # Find the model.pkl file in the downloaded structure
        # MLflow typically creates: dst_path/model/model.pkl for sklearn
        model_pkl = dst_path_obj / "model" / "model.pkl"
        if not model_pkl.exists():
            # Fallback: search for model.pkl recursively
            found_models = list(dst_path_obj.rglob("model.pkl"))
            if found_models:
                model_pkl = found_models[0]
                logger.info(f"Found model at: {model_pkl}")
            else:
                raise FileNotFoundError(
                    f"Model file (model.pkl) not found in downloaded "
                    f"structure at {dst_path}"
                )

        logger.info(
            f"Model successfully downloaded. "
            f"Model file located at: {model_pkl}"
        )
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()
