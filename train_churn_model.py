"""
train_churn_model.py

Telecom Customer Churn Prediction
Author: Mohammad Kasif Ansari

This script does the following:

1. Loads the telecom customer churn dataset from a CSV file.
2. Cleans the data (basic cleaning).
3. Splits the data into training and testing sets.
4. Builds two machine learning models:
      - Logistic Regression
      - Random Forest
5. Trains and evaluates both models.
6. Saves the best model to a file so it can be reused later.

To run this script:

    python train_churn_model.py

Make sure the file "TelecomCustomerChurn.csv" is in the same folder
as this script, or update the DATA_FILE path below.
"""

import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import (
    basic_data_cleaning,
    print_section,
)

# ---------------------------------------------------------
# CONFIGURATION – change these if your file/column is different
# ---------------------------------------------------------

# CSV file name (you can change the path if needed)
DATA_FILE = "TelecomCustomerChurn.csv"

# Name of the column that tells whether the customer churned
TARGET_COLUMN = "Churn"

# Folder where we will save the trained model
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)  # create folder if not present


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find dataset at {file_path}. "
            f"Please check the path or file name."
        )

    df = pd.read_csv(file_path)
    print(f" Loaded dataset from {file_path}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df


def get_feature_types(df: pd.DataFrame, target_col: str):
    """
    Figure out which columns are numeric and which are categorical.

    We simply look at the data type of each column.

    Returns
    -------
    numeric_features : list of str
    categorical_features : list of str
    """
    feature_columns = [c for c in df.columns if c != target_col]

    numeric_features = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col for col in feature_columns
        if col not in numeric_features
    ]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing object that:
      - scales numeric features
      - one-hot encodes categorical features
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_model_pipelines(preprocessor):
    """
    Create two machine learning pipelines:

    1. Logistic Regression
    2. Random Forest

    Each pipeline includes the same preprocessor so we don't have to
    repeat preprocessing code.
    """
    log_reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )),
        ]
    )

    models = {
        "Logistic Regression": log_reg_pipeline,
        "Random Forest": rf_pipeline,
    }

    return models


def prepare_target_column(y_raw: pd.Series) -> pd.Series:
    """
    Convert the target column into 0/1 values if needed.

    Many telecom churn datasets use "Yes"/"No" or similar labels.
    Here we try to map common text labels to numeric values.

    Anything that doesn't match will be mapped to 0 by default.
    """
    if y_raw.dtype != "object":
        # Already numeric, just return it
        return y_raw

    # Remove whitespace around values
    y_clean = y_raw.astype(str).str.strip()

    # Common mappings – you can adjust this if your dataset is different
    mapping = {
        "Yes": 1,
        "No": 0,
        "Churn": 1,
        "Not Churn": 0,
        "Stayed": 0,
        "Left": 1,
        "True": 1,
        "False": 0,
    }

    y_mapped = y_clean.map(mapping)

    # Any value that wasn't mapped will become NaN; we fill those with 0
    y_final = y_mapped.fillna(0).astype(int)

    return y_final


def evaluate_model(model_name: str, model, X_test, y_test) -> float:
    """
    Print evaluation metrics for a trained model and return its accuracy.
    """
    print_section(f"Evaluating {model_name}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy (simple and easy to understand)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Print more detailed metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def main():
    """
    Main function that ties everything together.
    This runs when you execute: python train_churn_model.py
    """
    print_section("Telecom Customer Churn – Training Started")

    # 1. Load dataset
    df = load_dataset(DATA_FILE)

    # 2. Basic data cleaning (remove duplicates, strip spaces, etc.)
    df = basic_data_cleaning(df)

    # 3. Check that the target column exists
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in dataset.\n"
            f"Available columns are: {list(df.columns)}"
        )

    # 4. Split into features (X) and target (y)
    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    # 5. Make sure the target is in numeric form (0/1)
    y = prepare_target_column(y_raw)

    # 6. Train/test split so we can fairly evaluate our model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,       # 20% for testing
        random_state=42,
        stratify=y,          # keep churn proportion similar in train/test
    )

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 7. Figure out which columns are numeric vs categorical
    numeric_features, categorical_features = get_feature_types(df, TARGET_COLUMN)
    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 8. Build preprocessor and models
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_model_pipelines(preprocessor)

    # 9. Train and evaluate each model, keep track of the best one
    best_model_name = None
    best_model = None
    best_accuracy = -1.0

    for name, model in models.items():
        print_section(f"Training {name}")
        model.fit(X_train, y_train)

        accuracy = evaluate_model(name, model, X_test, y_test)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    # 10. Show summary of the best model
    print_section("Best Model Summary")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    # 11. Save the best model to a file for later use
    model_path = os.path.join(MODEL_FOLDER, "best_churn_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n Saved best model to: {model_path}")

    print_section("Training Finished")


if __name__ == "__main__":
    main()
