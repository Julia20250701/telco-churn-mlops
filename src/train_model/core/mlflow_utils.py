from __future__ import annotations

import os
from pathlib import Path

import mlflow


def get_mlflow_mode() -> str:
    """
    Return the current MLflow mode.
    Supported:
    - local
    - dagshub
    """
    return os.getenv("MLFLOW_MODE", "local").strip().lower()


def setup_mlflow() -> str:
    """
    Configure MLflow using environment variables.
    Returns the experiment name in use.
    """
    mode = get_mlflow_mode()

    if mode == "dagshub":
        tracking_uri = os.getenv("DAGSHUB_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME_REMOTE",
            os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn"),
        )

        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not tracking_uri:
            raise ValueError(
                "DAGSHUB_MLFLOW_TRACKING_URI or MLFLOW_TRACKING_URI must be set for dagshub mode."
            )

        if not username or not password:
            raise ValueError(
                "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set "
                "for dagshub mode."
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn-local")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return experiment_name


def log_run_info(
    model_variant: str,
    model_type: str,
    data_path: str,
    train_rows: int,
    test_rows: int,
    metrics: dict,
    artifact_dir: str | Path,
) -> None:
    """
    Log run metadata, tags, metrics, and local artifacts.
    """
    artifact_dir = Path(artifact_dir)

    mlflow.log_param("model_variant", model_variant)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("train_rows", train_rows)
    mlflow.log_param("test_rows", test_rows)

    mlflow.set_tag("project", "telco-churn-mlops")
    mlflow.set_tag("stage", "training")
    mlflow.set_tag("artifact_dir", str(artifact_dir))
    mlflow.set_tag("mlflow_mode", get_mlflow_mode())

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1", metrics["f1"])

    churn_report = metrics.get("classification_report", {}).get("1", {})
    if "precision" in churn_report:
        mlflow.log_metric("churn_precision", churn_report["precision"])
    if "recall" in churn_report:
        mlflow.log_metric("churn_recall", churn_report["recall"])
    if "f1-score" in churn_report:
        mlflow.log_metric("churn_f1", churn_report["f1-score"])

    if artifact_dir.exists():
        mlflow.log_artifacts(str(artifact_dir), artifact_path="run_artifacts")
