from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.services.data_import.load_data import load_raw_data
from src.data.services.preprocess.preprocess import prepare_engineered_data
from src.evaluate_model.services.evaluate import compute_classification_metrics, save_metrics
from src.train_model.core.artifact_utils import create_run_dir, save_metadata


def train_engineered_model(data_path: str | Path) -> dict:
    """
    Train and evaluate the engineered-features churn model.
    Save all run artifacts into a timestamped run directory.
    """
    df = load_raw_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = prepare_engineered_data(df)

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

    metadata = {
        "model_variant": "engineered",
        "model_type": "LogisticRegression",
        "data_path": str(data_path),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "artifact_dir": str(run_dir),
    }
    save_metadata(metadata, metadata_path)

    print(f"Saved engineered model to: {model_path}")
    print(f"Saved engineered metrics to: {metrics_path}")
    print(f"Saved engineered metadata to: {metadata_path}")
    print(metrics)

    return metrics


if __name__ == "__main__":
    train_engineered_model("data/raw/telco_churn.csv")
