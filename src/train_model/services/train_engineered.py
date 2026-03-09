from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.services.data_import.load_data import load_raw_data
from src.data.services.preprocess.preprocess import prepare_engineered_data
from src.evaluate_model.services.evaluate import (
    compute_classification_metrics,
    save_metrics,
)
from src.train_model.core.artifact_utils import create_run_dir, save_metadata
from src.train_model.core.mlflow_utils import log_run_info, setup_mlflow


def train_engineered_model(data_path: str | Path) -> dict:
    """
    Train and evaluate the engineered churn model.
    Save local run artifacts and log the run to MLflow.
    """
    experiment_name = setup_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment  :", experiment_name)

    df = load_raw_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = prepare_engineered_data(df)

    with mlflow.start_run(run_name="engineered_logreg") as run:
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)

        run_dir = create_run_dir("engineered")
        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "metadata.json"

        joblib.dump(model, model_path)
        save_metrics(metrics, metrics_path)

        log_run_info(
            model_variant="engineered",
            model_type="LogisticRegression",
            data_path=str(data_path),
            train_rows=len(X_train),
            test_rows=len(X_test),
            metrics=metrics,
            artifact_dir=run_dir,
        )

        mlflow.log_param("feature_set", "engineered")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)
        mlflow.set_tag("selection_status", "experiment")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
        )

        metadata = {
            "model_variant": "engineered",
            "model_type": "LogisticRegression",
            "data_path": str(data_path),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "artifact_dir": str(run_dir),
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_uri": model_info.model_uri,
        }
        save_metadata(metadata, metadata_path)

        print(f"Saved engineered model to: {model_path}")
        print(f"Saved engineered metrics to: {metrics_path}")
        print(f"Saved engineered metadata to: {metadata_path}")
        print(f"MLflow run ID   : {run.info.run_id}")
        print(f"MLflow model URI: {model_info.model_uri}")
        print(metrics)

        return metrics


if __name__ == "__main__":
    train_engineered_model("data/raw/telco_churn.csv")
