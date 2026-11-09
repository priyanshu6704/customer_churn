
import joblib
import pandas as pd

from src.config import MODEL_PATHS, PREPROCESSOR_PATH


FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]



FEATURE_MAPPING = {
    "credit_score": "CreditScore",
    "country": "Geography",
    "gender": "Gender",
    "age": "Age",
    "tenure": "Tenure",
    "balance": "Balance",
    "products_number": "NumOfProducts",
    "credit_card": "HasCrCard",
    "active_member": "IsActiveMember",
    "estimated_salary": "EstimatedSalary"
}


def load_artifacts():

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    models = {
        "logistic_regression": joblib.load(
            MODEL_PATHS["Logistic Regression"]
        ),
        "random_forest": joblib.load(
            MODEL_PATHS["Random Forest"]
        ),
        "xgboost": joblib.load(
            MODEL_PATHS["XGBoost"]
        )
    }

    return preprocessor, models



preprocessor, models = load_artifacts()


def predict_all(customer_data):

    customer_data = {
        FEATURE_MAPPING[key]: value
        for key, value in customer_data.items()
    }

    input_df = pd.DataFrame(
        [customer_data],
        columns=FEATURE_COLUMNS
    )


    processed_input = preprocessor.transform(input_df)

    results = {}


    for model_name, model in models.items():

        prediction = int(model.predict(processed_input)[0])

        # Probability corresponding to the positive class (Exited = 1)
        positive_class_index = list(model.classes_).index(1)

        churn_probability = float(
            model.predict_proba(processed_input)[
                0, positive_class_index
            ]
        )

        results[model_name] = {
            "prediction": prediction,
            "churn_probability": round(churn_probability, 4),
            "result": (
                "Likely to Churn"
                if prediction == 1
                else "Unlikely to Churn"
            )
        }

    return results
