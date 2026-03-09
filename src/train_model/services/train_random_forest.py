from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.services.data_import.load_data import load_raw_data
from src.data.services.preprocess.preprocess import prepare_engineered_data
from src.evaluate_model.services.evaluate import compute_classification_metrics, save_metrics
from src.train_model.core.artifact_utils import create_run_dir, save_metadata
from src.train_model.core.mlflow_utils import log_run_info, setup_mlflow


def train_random_forest_model(data_path: str | Path) -> dict:
    """
    Train and evaluate a Random Forest churn model using engineered features.
    Save local run artifacts and log the run to MLflow.
    """
    setup_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    df = load_raw_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = prepare_engineered_data(df)

    n_estimators = 300
    max_depth = 12
    min_samples_split = 10
    min_samples_leaf = 4
    random_state = 42

    with mlflow.start_run(run_name="engineered_random_forest"):
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)

        run_dir = create_run_dir("random_forest")
        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "metadata.json"

        joblib.dump(model, model_path)
        save_metrics(metrics, metrics_path)

        metadata = {
            "model_variant": "engineered_random_forest",
            "model_type": "RandomForestClassifier",
            "data_path": str(data_path),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "artifact_dir": str(run_dir),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        }
        save_metadata(metadata, metadata_path)

        log_run_info(
            model_variant="engineered_random_forest",
            model_type="RandomForestClassifier",
            data_path=str(data_path),
            train_rows=len(X_train),
            test_rows=len(X_test),
            metrics=metrics,
            artifact_dir=run_dir,
        )

        mlflow.log_param("feature_set", "engineered")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_jobs", -1)

        mlflow.set_tag("selection_status", "experiment")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
        )

        print(f"Saved Random Forest model to: {model_path}")
        print(f"Saved Random Forest metrics to: {metrics_path}")
        print(f"Saved Random Forest metadata to: {metadata_path}")
        print(metrics)

        return metrics


if __name__ == "__main__":
    train_random_forest_model("data/raw/telco_churn.csv")
