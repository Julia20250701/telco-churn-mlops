# Manual workflow — current local MLOps pipeline

This document describes the current working manual pipeline before Airflow orchestration.

## 1. Train models

Run the training scripts:

```bash
uv run python -m src.train_model.services.train_baseline
uv run python -m src.train_model.services.train_engineered
uv run python -m src.train_model.services.train_random_forest
