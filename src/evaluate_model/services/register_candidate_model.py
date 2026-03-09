from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.train_model.core.mlflow_utils import setup_mlflow

COMPARISON_PATH = Path("results/model_comparison/latest_comparison.json")


def load_candidate_info() -> dict:
    """
    Load the selected winning model from the latest comparison file.
    """
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(f"Comparison file not found: {COMPARISON_PATH}")

    with COMPARISON_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    candidate = payload.get("candidate_model")
    if not candidate:
        raise ValueError("No candidate_model found in comparison report.")

    if not candidate.get("mlflow_run_id"):
        raise ValueError("Candidate model does not contain mlflow_run_id.")

    return candidate


def ensure_registered_model_exists(client: MlflowClient, model_name: str) -> None:
    """
    Ensure the registered model family exists.
    """
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            name=model_name,
            description="Registered model family for telco churn classification.",
        )


def wait_until_ready(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 60,
) -> None:
    """
    Wait until a model version becomes READY.
    """
    start_time = time.time()

    while True:
        model_version = client.get_model_version(model_name, version)
        status = model_version.status

        if status == ModelVersionStatus.to_string(ModelVersionStatus.READY):
            return

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(
                f"Model version {version} did not become READY within {timeout_seconds} seconds."
            )

        time.sleep(1)


def register_candidate_model(client: MlflowClient, candidate: dict) -> str:
    """
    Register the candidate model using the exact logged MLflow model URI if available.
    """
    model_name = os.getenv("REGISTERED_MODEL_NAME", "telco-churn-classifier")

    model_uri = candidate.get("mlflow_model_uri")
    if not model_uri:
        run_id = candidate["mlflow_run_id"]
        model_uri = f"runs:/{run_id}/model"

    print("Selected candidate:")
    print(f"  model_variant : {candidate['model_variant']}")
    print(f"  run_id        : {candidate['mlflow_run_id']}")
    print(f"  model_uri     : {model_uri}")

    ensure_registered_model_exists(client, model_name)

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(result.version)


def main() -> None:
    setup_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    client = MlflowClient()

    model_name = os.getenv("REGISTERED_MODEL_NAME", "telco-churn-classifier")
    model_alias = os.getenv("MODEL_ALIAS", "candidate")

    candidate = load_candidate_info()
    version = register_candidate_model(client, candidate)

    wait_until_ready(client, model_name, version)
    client.set_registered_model_alias(model_name, model_alias, version)

    print(f"Registered model name : {model_name}")
    print(f"Registered version    : {version}")
    print(f"Alias set             : {model_alias}")


if __name__ == "__main__":
    main()
