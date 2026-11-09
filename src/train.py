
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    MODEL_DIR,
    MODEL_PATHS,
    RANDOM_STATE
)

from src.preprocess import load_preprocessed_data


def initialize_models():
    """Initialize all three classification models."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


def train_all_models(data=None, save_models=True):
    """Train all three models and optionally save them."""

    if data is None:
        data = load_preprocessed_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    models = initialize_models()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        print(f"{name} training completed!")

        if save_models:
            model_path = MODEL_PATHS[name]

            joblib.dump(model, model_path)

            print(f"Model saved: {model_path}")

    return models


if __name__ == "__main__":

    trained_models = train_all_models()

    print("\nAll models trained successfully!")
