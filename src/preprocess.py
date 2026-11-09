
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    RANDOM_STATE,
    VALIDATION_SIZE,
    TEST_SIZE,
    PREPROCESSOR_PATH
)


def load_dataset():
    """Load the customer churn dataset."""

    return pd.read_csv(DATASET_PATH)


def split_data(df):
    """Split data into stratified train, validation and test sets."""

    X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    y = df[TARGET_COLUMN]

    # First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split: 15% validation, 15% testing
    test_ratio = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessor():
    """Load the previously fitted preprocessing pipeline."""

    return joblib.load(PREPROCESSOR_PATH)


def load_preprocessed_data():
    """Load, split and transform the complete dataset."""

    df = load_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(df)
    )

    preprocessor = load_preprocessor()

    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Data preprocessing completed!")
    print("Training:", X_train_processed.shape)
    print("Validation:", X_val_processed.shape)
    print("Testing:", X_test_processed.shape)

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor
    }
