
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Dataset configuration
DATASET_PATH = DATA_DIR / "bank_churn.csv"

TARGET_COLUMN = "Exited"

DROP_COLUMNS = [
    "id",
    "CustomerId",
    "Surname"
]

# Data splitting configuration
RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Preprocessor artifact
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# Model artifact paths
LOGISTIC_REGRESSION_PATH = (
    MODEL_DIR / "logistic_regression.joblib"
)

RANDOM_FOREST_PATH = (
    MODEL_DIR / "random_forest.joblib"
)

XGBOOST_PATH = MODEL_DIR / "xgboost.joblib"

MODEL_PATHS = {
    "Logistic Regression": LOGISTIC_REGRESSION_PATH,
    "Random Forest": RANDOM_FOREST_PATH,
    "XGBoost": XGBOOST_PATH
}

# MLflow configuration
EXPERIMENT_NAME = "customer_churn"

MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)
