from __future__ import annotations

import os
from typing import Optional

import mlflow


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_mlflow_mode() -> str:
    return os.getenv("MLFLOW_MODE", "local").strip().lower()


def get_tracking_uri() -> str:
    """
    Return the MLflow tracking URI based on the selected mode.

    Supported modes:
    - local
    - dagshub
    """
    mode = _normalize_mlflow_mode()

    if mode == "local":
        return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    if mode == "dagshub":
        return _get_env("DAGSHUB_MLFLOW_TRACKING_URI")

    raise ValueError(f"Unsupported MLFLOW_MODE='{mode}'. Use 'local' or 'dagshub'.")


def configure_mlflow_tracking() -> str:
    """
    Configure only the MLflow tracking connection.

    This function is safe for both training and serving because it does NOT:
    - create/set experiments
    - assume training behavior
    - start runs

    It only points the client to the correct tracking server.
    """
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def setup_mlflow_for_training() -> str:
    """
    Configure MLflow for training workflows.

    This:
    - sets the tracking URI
    - sets the experiment name

    Use this in training / evaluation scripts that log runs.
    """
    tracking_uri = configure_mlflow_tracking()
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        os.getenv("MLFLOW_EXPERIMENT_NAME_REMOTE", "telco-churn-local"),
    )
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def setup_mlflow_for_serving() -> str:
    """
    Configure MLflow for serving workflows.

    This:
    - sets the tracking URI
    - does NOT set the experiment

    Use this in prediction APIs or other model-loading code.
    """
    tracking_uri = configure_mlflow_tracking()
    return tracking_uri


def get_registered_model_name() -> str:
    return _get_env("REGISTERED_MODEL_NAME", "telco-churn-classifier")


def get_model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "candidate")


def get_registry_model_uri() -> str:
    """
    Example:
    models:/telco-churn-classifier@candidate
    """
    model_name = get_registered_model_name()
    model_alias = get_model_alias()
    return f"models:/{model_name}@{model_alias}"
