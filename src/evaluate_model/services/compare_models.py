from __future__ import annotations

import json
from pathlib import Path


def get_latest_run_dir(model_variant: str) -> Path:
    """
    Return the latest run directory for a given model variant.
    Example:
    artifacts/baseline/20260306_104405/
    """
    base_dir = Path("artifacts") / model_variant

    if not base_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {base_dir}")

    run_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {base_dir}")

    latest_run = sorted(run_dirs)[-1]
    return latest_run


def load_metrics(path: str | Path) -> dict:
    """
    Load a metrics JSON file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_summary(metrics: dict) -> dict:
    """
    Extract the most important metrics for quick comparison.
    """
    churn_report = metrics.get("classification_report", {}).get("1", {})

    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "churn_precision": churn_report.get("precision"),
        "churn_recall": churn_report.get("recall"),
        "churn_f1": churn_report.get("f1-score"),
    }


def print_comparison_table(baseline: dict, engineered: dict) -> None:
    """
    Print a simple side-by-side comparison table.
    """
    print("\n--- Latest Model Comparison ---")
    print(f"{'Metric':<18} {'Baseline':<12} {'Engineered':<12}")
    print("-" * 44)

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "churn_precision",
        "churn_recall",
        "churn_f1",
    ]:
        baseline_value = baseline.get(key)
        engineered_value = engineered.get(key)

        baseline_str = f"{baseline_value:.4f}" if baseline_value is not None else "N/A"
        engineered_str = f"{engineered_value:.4f}" if engineered_value is not None else "N/A"

        print(f"{key:<18} {baseline_str:<12} {engineered_str:<12}")


def choose_best_model(baseline: dict, engineered: dict, metric: str = "churn_f1") -> str:
    """
    Choose the better model based on a chosen metric.
    Default: churn_f1 because churn detection is the business focus.
    """
    baseline_value = baseline.get(metric)
    engineered_value = engineered.get(metric)

    if baseline_value is None or engineered_value is None:
        return f"Could not determine best model because metric '{metric}' is missing."

    if baseline_value > engineered_value:
        return f"Baseline model is better based on {metric}: {baseline_value:.4f} > {engineered_value:.4f}"
    if engineered_value > baseline_value:
        return f"Engineered model is better based on {metric}: {engineered_value:.4f} > {baseline_value:.4f}"

    return f"Both models are tied on {metric}: {baseline_value:.4f}"


def main() -> None:
    baseline_run_dir = get_latest_run_dir("baseline")
    engineered_run_dir = get_latest_run_dir("engineered")

    baseline_metrics = load_metrics(baseline_run_dir / "metrics.json")
    engineered_metrics = load_metrics(engineered_run_dir / "metrics.json")

    baseline_summary = extract_summary(baseline_metrics)
    engineered_summary = extract_summary(engineered_metrics)

    print(f"Latest baseline run:   {baseline_run_dir}")
    print(f"Latest engineered run: {engineered_run_dir}")

    print_comparison_table(baseline_summary, engineered_summary)
    print()
    print(choose_best_model(baseline_summary, engineered_summary, metric="churn_f1"))


if __name__ == "__main__":
    main()
