from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results/model_comparison")


def get_latest_run_dir(model_dir: Path) -> Path | None:
    """
    Return the latest timestamped run directory inside a model artifact folder.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    return sorted(run_dirs)[-1]


def load_json(path: Path) -> dict:
    """
    Load a JSON file from disk.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_relevant_metrics(metrics: dict) -> dict:
    """
    Extract the main metrics used for comparison.
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


def collect_model_results() -> tuple[list[dict], list[dict]]:
    """
    Collect the latest metrics and metadata for each model folder in artifacts/.

    Returns:
        valid_results: models that have metrics + metadata + mlflow_run_id
        skipped_results: models skipped because something important is missing
    """
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError("artifacts/ directory not found.")

    valid_results: list[dict] = []
    skipped_results: list[dict] = []

    for model_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        latest_run_dir = get_latest_run_dir(model_dir)
        if latest_run_dir is None:
            skipped_results.append(
                {
                    "model_variant": model_dir.name,
                    "run_dir": None,
                    "reason": "no run directories found",
                }
            )
            continue

        metrics_path = latest_run_dir / "metrics.json"
        metadata_path = latest_run_dir / "metadata.json"

        if not metrics_path.exists():
            skipped_results.append(
                {
                    "model_variant": model_dir.name,
                    "run_dir": str(latest_run_dir),
                    "reason": "metrics.json missing",
                }
            )
            continue

        if not metadata_path.exists():
            skipped_results.append(
                {
                    "model_variant": model_dir.name,
                    "run_dir": str(latest_run_dir),
                    "reason": "metadata.json missing",
                }
            )
            continue

        metrics = load_json(metrics_path)
        metadata = load_json(metadata_path)

        mlflow_run_id = metadata.get("mlflow_run_id")
        mlflow_model_uri = metadata.get("mlflow_model_uri")

        if not mlflow_run_id:
            skipped_results.append(
                {
                    "model_variant": model_dir.name,
                    "run_dir": str(latest_run_dir),
                    "reason": "mlflow_run_id missing in metadata.json",
                }
            )
            continue

        extracted = extract_relevant_metrics(metrics)

        valid_results.append(
            {
                "model_variant": model_dir.name,
                "run_dir": str(latest_run_dir),
                "mlflow_run_id": mlflow_run_id,
                "mlflow_model_uri": mlflow_model_uri,
                **extracted,
            }
        )

    return valid_results, skipped_results


def rank_results(results: list[dict]) -> list[dict]:
    """
    Rank models using the business rule:
    1. churn_f1
    2. churn_recall
    3. precision
    Higher is better.
    """
    return sorted(
        results,
        key=lambda x: (
            x.get("churn_f1") or 0.0,
            x.get("churn_recall") or 0.0,
            x.get("precision") or 0.0,
        ),
        reverse=True,
    )


def format_metric(value: float | None) -> str:
    """
    Format metric values nicely for printing.
    """
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def save_comparison_report(ranked: list[dict], skipped_results: list[dict]) -> tuple[Path, Path]:
    """
    Save the ranked comparison report to:
    - a timestamped JSON file
    - a stable latest JSON file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = RESULTS_DIR / f"{timestamp}_comparison.json"
    latest_path = RESULTS_DIR / "latest_comparison.json"

    winner = ranked[0] if ranked else None

    payload = {
        "generated_at": timestamp,
        "selection_rule": [
            "best churn_f1",
            "then best churn_recall",
            "then best precision",
        ],
        "candidate_model": winner,
        "ranked_results": ranked,
        "skipped_models": skipped_results,
    }

    with timestamped_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return timestamped_path, latest_path


def print_report(results: list[dict], skipped_results: list[dict]) -> None:
    """
    Print a ranked comparison report.
    """
    print("\n=== MODEL COMPARISON REPORT ===\n")
    print("Selection rule:")
    print("1. Best churn_f1")
    print("2. Then best churn_recall")
    print("3. Then best precision\n")

    if not results:
        print("No valid model results found.\n")
    else:
        ranked = rank_results(results)
        winner = ranked[0]

        for idx, result in enumerate(ranked, start=1):
            print(f"Rank #{idx}: {result['model_variant']}")
            print(f"  run_dir          : {result['run_dir']}")
            print(f"  mlflow_run_id    : {result['mlflow_run_id']}")
            print(f"  mlflow_model_uri : {result.get('mlflow_model_uri')}")
            print(f"  accuracy         : {format_metric(result['accuracy'])}")
            print(f"  precision        : {format_metric(result['precision'])}")
            print(f"  recall           : {format_metric(result['recall'])}")
            print(f"  f1               : {format_metric(result['f1'])}")
            print(f"  churn_precision  : {format_metric(result['churn_precision'])}")
            print(f"  churn_recall     : {format_metric(result['churn_recall'])}")
            print(f"  churn_f1         : {format_metric(result['churn_f1'])}")
            print()

        print("=== CURRENT BEST CANDIDATE ===\n")
        print(f"model_variant   : {winner['model_variant']}")
        print(f"run_dir         : {winner['run_dir']}")
        print(f"mlflow_run_id   : {winner['mlflow_run_id']}")
        print(f"mlflow_model_uri: {winner.get('mlflow_model_uri')}")
        print(f"churn_f1        : {format_metric(winner['churn_f1'])}")
        print(f"churn_recall    : {format_metric(winner['churn_recall'])}")
        print(f"precision       : {format_metric(winner['precision'])}")
        print()

    if skipped_results:
        print("=== SKIPPED MODELS ===\n")
        for skipped in skipped_results:
            print(f"model_variant : {skipped['model_variant']}")
            print(f"run_dir       : {skipped['run_dir']}")
            print(f"reason        : {skipped['reason']}")
            print()


def main() -> None:
    results, skipped_results = collect_model_results()
    print_report(results, skipped_results)

    if results:
        ranked = rank_results(results)
        timestamped_path, latest_path = save_comparison_report(ranked, skipped_results)
        print("Saved comparison report files:")
        print(f"  {timestamped_path}")
        print(f"  {latest_path}")


if __name__ == "__main__":
    main()
