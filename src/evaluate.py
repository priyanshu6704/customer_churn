
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X, y):
    """Calculate classification evaluation metrics."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = {
        "ROC-AUC": roc_auc_score(y, probabilities),
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y, predictions, zero_division=0
        )
    }

    return results


def evaluate_all_models(models, data, split="test"):
    """Evaluate all models on validation or test data."""

    if split not in ("validation", "test"):
        raise ValueError(
            "split must be either 'validation' or 'test'"
        )

    X = data["X_val"] if split == "validation" else data["X_test"]
    y = data["y_val"] if split == "validation" else data["y_test"]

    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(model, X, y)

    return results


def compare_models(results):
    """Create a model comparison DataFrame."""

    comparison = pd.DataFrame.from_dict(
        results,
        orient="index"
    )

    comparison.index.name = "Model"

    return comparison.reset_index()


def generate_confusion_matrix(model, X, y):
    """Generate confusion matrix for a model."""

    predictions = model.predict(X)

    return confusion_matrix(y, predictions)


if __name__ == "__main__":

    import joblib

    from src.config import MODEL_PATHS
    from src.preprocess import load_preprocessed_data

    data = load_preprocessed_data()

    models = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }

    # Validation evaluation
    val_results = evaluate_all_models(
        models, data, split="validation"
    )

    print("\nVALIDATION RESULTS")
    print(compare_models(val_results).round(4))

    # Final test evaluation
    test_results = evaluate_all_models(
        models, data, split="test"
    )

    print("\nTEST RESULTS")
    print(compare_models(test_results).round(4))
