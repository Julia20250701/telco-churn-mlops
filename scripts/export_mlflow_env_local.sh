#!/usr/bin/env bash
set -a
source .env
set +a

export MLFLOW_MODE=local
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}"

echo "MLflow mode        : ${MLFLOW_MODE}"
echo "Tracking URI       : ${MLFLOW_TRACKING_URI}"
echo "Experiment name    : ${MLFLOW_EXPERIMENT_NAME}"
