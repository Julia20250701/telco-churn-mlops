from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn
from dotenv import load_dotenv

from src.train_model.core.mlflow_utils import setup_mlflow

# Make sure .env is loaded when this module is imported
load_dotenv(override=False)


@lru_cache(maxsize=1)
def load_prediction_model() -> Any:
    """
    Load and cache the prediction model.

    If PREDICT_LOAD_FROM_REGISTRY=1:
        load from MLflow Model Registry using model alias.
    Otherwise:
        load from a local joblib file.
    """
    load_from_registry = os.getenv("PREDICT_LOAD_FROM_REGISTRY", "0") == "1"

    if load_from_registry:
        setup_mlflow()

        model_name = os.getenv("REGISTERED_MODEL_NAME", "telco-churn-classifier")
        model_alias = os.getenv("MODEL_ALIAS", "candidate")
        model_uri = f"models:/{model_name}@{model_alias}"

        print(f"Loading prediction model from MLflow Registry: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)

    local_model_path = Path(
        os.getenv(
            "LOCAL_MODEL_PATH",
            "artifacts/baseline/latest/model.joblib",
        )
    )

    if not local_model_path.exists():
        raise FileNotFoundError(f"Local model file not found: {local_model_path}")

    print(f"Loading prediction model from local path: {local_model_path}")
    return joblib.load(local_model_path)


def get_model_source() -> str:
    """
    Return the active model source label for API responses.
    """
    load_from_registry = os.getenv("PREDICT_LOAD_FROM_REGISTRY", "0") == "1"
    return "registry" if load_from_registry else "local"


def clear_model_cache() -> None:
    """
    Clear the cached model.
    Useful for tests or manual model reloads.
    """
    load_prediction_model.cache_clear()
