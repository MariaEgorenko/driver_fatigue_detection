import logging
import mlflow
from mlflow.tracking import MlflowClient

from backend.app.core.config import get_settings, Settings
from ml.src.inference.video_pipeline import VideoPipeline

logger = logging.getLogger(__name__)

def load_model_from_registry(
    model_name: str = "fatigue_pipeline",
    stage: str = "Production",
) -> VideoPipeline:
    """
    Download the pipeline from the MLflow Model Registry.
    If an error occurs, it returns a VideoPipeline with standard settings.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])

        if not versions:
            raise ValueError(f"No {stage} version for {model_name}")

        version = versions[0]
        run = client.get_run(version.run_id)
        params = run.data.params  # Parameters from MLflow (all come as strings)

        valid_setting_keys = set(Settings.model_fields.keys())
        overrides = {}

        for param_key, param_value in params.items():
            if param_key in valid_setting_keys:
                overrides[param_key] = param_value

        merged_data = settings.model_dump()
        merged_data.update(overrides)

        # согласно аннотациям в классе Settings!
        mlflow_settings = Settings.model_validate(merged_data)

        pipeline = VideoPipeline(mlflow_settings)
        logger.info(f"Loaded model {model_name} v{version.version} from MLflow ({stage})")
        return pipeline

    except Exception as e:
        logger.warning(f"MLflow load failed: {e}. Using default config.")
        return VideoPipeline(settings)
