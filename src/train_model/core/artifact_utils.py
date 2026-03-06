from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def get_run_timestamp() -> str:
    """
    Return a UTC timestamp string for artifact folders.
    Example: 20260306_105800
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def create_run_dir(model_variant: str) -> Path:
    """
    Create and return a run directory like:
    artifacts/baseline/20260306_105800/
    """
    timestamp = get_run_timestamp()
    run_dir = Path("artifacts") / model_variant / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_metadata(metadata: dict, output_path: str | Path) -> None:
    """
    Save metadata as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
