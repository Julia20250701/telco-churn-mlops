from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.train_model.core.mlflow_utils import setup_mlflow

COMPARISON_PATH = Path("results/model_comparison/latest_comparison.json")


def load_candidate_run_id() -> str:
    """
    Load the winning MLflow run ID from the latest comparison report.
    """
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(f"Comparison file not found: {COMPARISON_PATH}")

    with COMPARISON_PATH.open("r", encoding="utf-8") as f:
        comparison = json.load(f)

    candidate = comparison.get("candidate_model")
    if not candidate:
        raise ValueError("No candidate_model found in comparison report.")

    run_id = candidate.get("mlflow_run_id")
    if not run_id:
        raise ValueError("No mlflow_run_id found for candidate model.")

    return run_id


def clear_existing_candidate_tags(client: MlflowClient, experiment_name: str) -> None:
    """
    Change any existing candidate-tagged runs to previous_candidate.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.selection_status = 'candidate'",
    )

    for run in runs:
        client.set_tag(
            run.info.run_id,
            "selection_status",
            "previous_candidate",
        )


def tag_new_candidate(client: MlflowClient, run_id: str) -> None:
    """
    Tag the selected run as the current candidate.
    """
    client.set_tag(run_id, "selection_status", "candidate")


def main() -> None:
    experiment_name = setup_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    client = MlflowClient()
    candidate_run_id = load_candidate_run_id()

    clear_existing_candidate_tags(client, experiment_name)
    tag_new_candidate(client, candidate_run_id)

    print(f"Tagged MLflow run as candidate: {candidate_run_id}")


if __name__ == "__main__":
    main()
