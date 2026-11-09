
import joblib
import mlflow

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATHS,
    PREPROCESSOR_PATH
)

from src.preprocess import load_preprocessed_data

from src.evaluate import evaluate_all_models


def load_trained_models():
    """Load all previously trained models."""

    models = {}

    for name, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}"
            )

        models[name] = joblib.load(path)

    return models


def log_model_parameters(model):
    """Log model hyperparameters."""

    params = model.get_params(deep=False)

    mlflow.log_params({
        key: str(value)
        for key, value in params.items()
    })


def log_model_metrics(model, data):
    """Log validation and test metrics."""

    validation_results = evaluate_all_models(
        {"model": model},
        data,
        split="validation"
    )["model"]

    test_results = evaluate_all_models(
        {"model": model},
        data,
        split="test"
    )["model"]

    # Log validation metrics
    mlflow.log_metrics({
        f"val_{key.lower().replace('-', '_')}": float(value)
        for key, value in validation_results.items()
    })

    # Log test metrics
    mlflow.log_metrics({
        f"test_{key.lower().replace('-', '_')}": float(value)
        for key, value in test_results.items()
    })

    return validation_results, test_results


def track_experiments():
    """Track all three models as independent MLflow runs."""

    # Configure tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load existing artifacts
    models = load_trained_models()

    data = load_preprocessed_data()

    for name, model in models.items():

        print(f"\nTracking experiment: {name}")

        with mlflow.start_run(run_name=name):

            # Log model information
            mlflow.log_param("model_name", name)

            log_model_parameters(model)

            # Log validation and test metrics
            val_metrics, test_metrics = log_model_metrics(
                model, data
            )

            # Log trained model artifact
            mlflow.log_artifact(
                str(MODEL_PATHS[name]),
                artifact_path="model"
            )

            # Log fitted preprocessing pipeline
            mlflow.log_artifact(
                str(PREPROCESSOR_PATH),
                artifact_path="preprocessing"
            )

            print("Validation:", val_metrics)
            print("Test:", test_metrics)

            print(f"{name} logged successfully!")

    print("\nAll three MLflow experiments completed!")


if __name__ == "__main__":
    track_experiments()
