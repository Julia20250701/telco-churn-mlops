from __future__ import annotations

from fastapi.testclient import TestClient

from src.predict.services.predict_api import app


class DummyModel:
    def predict(self, df):
        return [1]

    def predict_proba(self, df):
        return [[0.2, 0.8]]


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "predict api running"}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_mocked_model(monkeypatch) -> None:
    monkeypatch.setattr(app.state, "model", DummyModel(), raising=False)
    monkeypatch.setenv("PREDICT_LOAD_FROM_REGISTRY", "1")

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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == 1
    assert data["prediction_label"] == "churn"
    assert data["churn_probability"] == 0.8
    assert data["model_source"] == "registry"


def test_reload_model_with_mocked_loader(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.predict.services.predict_api.load_prediction_model",
        lambda: DummyModel(),
    )
    monkeypatch.setenv("PREDICT_LOAD_FROM_REGISTRY", "1")

    response = client.post("/reload-model")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["model_source"] == "registry"
    assert data["message"] == "Model cache cleared and model reloaded successfully."
