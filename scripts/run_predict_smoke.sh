#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

uv run pytest -q tests/smoke/test_predict_api_smoke.py
