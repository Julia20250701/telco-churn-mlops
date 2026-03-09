from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.predict.core.load_model import (
    clear_model_cache,
    get_model_source,
    load_prediction_model,
)
from src.predict.core.schemas import PredictionRequest, PredictionResponse

# Explicitly load .env before app startup logic runs
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DEBUG: app startup - loading model...")
    try:
        app.state.model = load_prediction_model()
        print(f"DEBUG: app startup - model loaded -> {type(app.state.model)}")
    except Exception as exc:
        print(f"DEBUG: app startup - model load failed -> {exc}")
        raise
    yield


app = FastAPI(title="Predict API", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "predict api running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reload-model")
def reload_model() -> dict[str, str]:
    try:
        clear_model_cache()
        app.state.model = load_prediction_model()
        return {
            "status": "ok",
            "model_source": get_model_source(),
            "message": "Model cache cleared and model reloaded successfully.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = app.state.model

        features_df = pd.DataFrame([payload.model_dump()])

        prediction = int(model.predict(features_df)[0])

        churn_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_df)
            churn_probability = float(probabilities[0][1])

        prediction_label = "churn" if prediction == 1 else "no churn"

        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            churn_probability=churn_probability,
            model_source=get_model_source(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
