from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from src.predict.core.load_model import load_prediction_model
from src.predict.core.schemas import HealthResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DEBUG: app startup - loading model...")
    app.state.model = load_prediction_model()
    print(f"DEBUG: app startup - model loaded -> {type(app.state.model)}")
    yield


app = FastAPI(
    title="Telco Churn Predict API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = app.state.model

    input_df = pd.DataFrame([payload.model_dump()])

    prediction = model.predict(input_df)[0]

    churn_probability = None
    if hasattr(model, "predict_proba"):
        churn_probability = float(model.predict_proba(input_df)[0][1])

    prediction_int = int(prediction)
    prediction_label = "churn" if prediction_int == 1 else "no_churn"

    model_source = (
        "registry" if os.getenv("PREDICT_LOAD_FROM_REGISTRY", "0") == "1" else "local_file"
    )

    return PredictionResponse(
        prediction=prediction_int,
        prediction_label=prediction_label,
        churn_probability=churn_probability,
        model_source=model_source,
    )
