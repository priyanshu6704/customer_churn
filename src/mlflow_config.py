
import mlflow
from pathlib import Path

# 1. Identify project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Configure SQLite database for MLflow
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)

# 3. Define MLflow experiment
EXPERIMENT_NAME = "customer_churn"

# 4. Create or retrieve experiment
experiment = mlflow.set_experiment(EXPERIMENT_NAME)


print("\nMLflow Configuration Successful!")

print(f"Project Directory : {BASE_DIR}")
print(f"Tracking URI      : {mlflow.get_tracking_uri()}")
print(f"Experiment Name   : {EXPERIMENT_NAME}")
print(f"Experiment ID     : {experiment.experiment_id}")

