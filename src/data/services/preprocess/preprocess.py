from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco churn dataset.

    Steps:
    - drop customerID if present
    - convert TotalCharges to numeric
    - impute missing TotalCharges with the median
    - map Churn to binary
    - drop rows with missing target
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features based on the notebook logic.
    """
    df = df.copy()

    if "tenure" in df.columns:
        bins = [0, 12, 24, 48, 60, 72]
        labels = ["0-1 Year", "1-2 Years", "2-4 Years", "4-5 Years", "5+ Years"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    internet_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in internet_service_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No"})

    existing_service_cols = [col for col in internet_service_cols if col in df.columns]
    if existing_service_cols:
        df["num_add_services"] = (df[existing_service_cols] == "Yes").sum(axis=1)

    if {"MonthlyCharges", "tenure"}.issubset(df.columns):
        df["monthly_charge_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split cleaned dataframe into features X and target y.
    """
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split with stratification.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return numerical and categorical feature names.
    """
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numerical_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - scale numerical columns
    - one-hot encode categorical columns
    """
    numerical_features, categorical_features = get_feature_types(X)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def prepare_baseline_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Prepare cleaned baseline data and corresponding preprocessor.
    """
    df_clean = clean_telco_data(df)
    X, y = split_features_target(df_clean)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


def prepare_engineered_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Prepare cleaned + engineered data and corresponding preprocessor.
    """
    df_clean = clean_telco_data(df)
    df_eng = add_engineered_features(df_clean)
    X, y = split_features_target(df_eng)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    from src.data.services.data_import.load_data import load_raw_data

    df_raw = load_raw_data("data/raw/telco_churn.csv")

    X_train_base, X_test_base, y_train_base, y_test_base, _ = prepare_baseline_data(df_raw)
    print("Baseline X_train shape:", X_train_base.shape)
    print("Baseline X_test shape:", X_test_base.shape)

    X_train_eng, X_test_eng, y_train_eng, y_test_eng, _ = prepare_engineered_data(df_raw)
    print("Engineered X_train shape:", X_train_eng.shape)
    print("Engineered X_test shape:", X_test_eng.shape)
