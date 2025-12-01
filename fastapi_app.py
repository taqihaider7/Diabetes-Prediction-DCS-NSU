from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/diabetes_rf.pkl")

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

model = None
feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class PatientFeatures(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `dvc repro` or `python src/train.py` first to train the model."
        )
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Diabetes Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_diabetes(features: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert request to DataFrame with the correct column order
    data = pd.DataFrame([[getattr(features, col) for col in feature_columns]],
                        columns=feature_columns)

    prob = float(model.predict_proba(data)[0, 1])
    pred = int(prob >= 0.5)

    return PredictionResponse(prediction=pred, probability=prob)
