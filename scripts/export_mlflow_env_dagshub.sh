#!/usr/bin/env bash
set -a
source .env
set +a

export MLFLOW_MODE=dagshub
export MLFLOW_TRACKING_URI="${DAGSHUB_MLFLOW_TRACKING_URI}"
export MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME}"
export MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME_REMOTE}"

echo "MLflow mode        : ${MLFLOW_MODE}"
echo "Tracking URI       : ${MLFLOW_TRACKING_URI}"
echo "Experiment name    : ${MLFLOW_EXPERIMENT_NAME}"
echo "Tracking user      : ${MLFLOW_TRACKING_USERNAME}"
