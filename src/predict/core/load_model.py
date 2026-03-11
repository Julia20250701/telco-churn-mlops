from __future__ import annotations

import os

import joblib
import mlflow.pyfunc

from src.train_model.core.mlflow_utils import (
    get_registry_model_uri,
    setup_mlflow_for_serving,
)


def _load_from_registry():
    model_uri = get_registry_model_uri()
    print(f"Loading prediction model from MLflow Registry: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def _load_from_local_file():
    local_model_path = os.getenv("PREDICT_LOCAL_MODEL_PATH", "").strip()
    if not local_model_path:
        raise ValueError(
            "PREDICT_LOCAL_MODEL_PATH must be set when "
            "PREDICT_LOAD_FROM_REGISTRY is not enabled."
        )

    print(f"Loading prediction model from local file: {local_model_path}")
    return joblib.load(local_model_path)


def load_prediction_model():
    """
    Load the prediction model either:
    - from the MLflow Model Registry
    - or from a local joblib file

    Registry mode is enabled with:
    PREDICT_LOAD_FROM_REGISTRY=1
    """
    load_from_registry = os.getenv("PREDICT_LOAD_FROM_REGISTRY", "0") == "1"

    if load_from_registry:
        setup_mlflow_for_serving()
        return _load_from_registry()

    return _load_from_local_file()
