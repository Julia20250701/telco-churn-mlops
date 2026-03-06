from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Compute standard classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    """
    Save metrics as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    example_metrics = {
        "accuracy": 0.8,
        "precision": 0.7,
        "recall": 0.6,
        "f1": 0.65,
    }
    save_metrics(example_metrics, "results/example_metrics.json")
    print("Saved example metrics.")
