from __future__ import annotations

import os

import requests

BASE_URL = os.getenv("PREDICT_API_BASE_URL", "http://127.0.0.1:8000")


def test_health_smoke() -> None:
    response = requests.get(f"{BASE_URL}/health", timeout=10)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_smoke() -> None:
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.2,
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=20,
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "prediction_label" in body
    assert "model_source" in body

    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["churn", "no churn"]
    assert body["model_source"] == "registry"

    if "churn_probability" in body and body["churn_probability"] is not None:
        assert 0.0 <= body["churn_probability"] <= 1.0
