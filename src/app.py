
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import uvicorn
import mlflow
import mlflow.sklearn

# DAGsHub configuration for tracking predictions
DAGSHUB_USERNAME = "nawazishpatana"
DAGSHUB_TOKEN = "a3e0c3abf610a86cea1f349c92b841da5816eac3"
DAGSHUB_TRACKING_URI = "https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow"

# Setup MLflow for prediction tracking
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
mlflow.set_experiment("Diabetes-Prediction")

# Global variables for model and scaler
model = None
scaler = None
model_metadata = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, model_metadata
    
    try:
        model_path = "models/best_diabetes_model.pkl"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}. Please train the model first.")
            return
        
        print(f"🔍 Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            model_metadata = model_data.get('metadata', {})
        
        if model is None:
            print("❌ Model is None after loading")
            return
            
        print("✅ Model loaded successfully")
        print(f"📊 Model type: {type(model).__name__}")
        if model_metadata:
            print(f"🎯 Model name: {model_metadata.get('model_name', 'N/A')}")
            print(f"📈 Model accuracy: {model_metadata.get('accuracy', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Diabetes Prediction API...")
    load_model()
    yield
    # Shutdown
    print("👋 Shutting down Diabetes Prediction API...")

# FastAPI app with lifespan
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes based on health metrics using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Pydantic models for request/response
class DiabetesFeatures(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 30,
                "Insulin": 100,
                "BMI": 25.5,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 35
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str
    confidence: str
    model_used: str

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]

@app.get("/")
async def root():
    return {"message": "Diabetes Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if model is not None else "model not loaded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_metadata": model_metadata,
        "features_used": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: DiabetesFeatures):
    """Make a single prediction and track it in MLflow"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert features to array
        feature_array = np.array([[
            features.Pregnancies,
            features.Glucose,
            features.BloodPressure,
            features.SkinThickness,
            features.Insulin,
            features.BMI,
            features.DiabetesPedigreeFunction,
            features.Age
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        # Determine confidence level
        prob = probability[1] if prediction == 1 else probability[0]
        if prob > 0.8:
            confidence = "high"
        elif prob > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Log prediction to MLflow
        with mlflow.start_run(run_name=f"prediction_{np.random.randint(1000)}"):
            mlflow.log_param("user", "api_user")
            mlflow.log_param("prediction", int(prediction))
            mlflow.log_metric("probability", float(probability[1]))
            mlflow.log_metric("confidence_score", prob)
            mlflow.log_dict(features.dict(), "input_features.json")
            
            # Log all input features as parameters
            for key, value in features.dict().items():
                mlflow.log_param(key, value)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability[1]),
            message="Diabetic" if prediction == 1 else "Not Diabetic",
            confidence=confidence,
            model_used=model_metadata.get('model_name', 'Unknown')
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[DiabetesFeatures]):
    """Make batch predictions"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to array
        feature_arrays = []
        for features in features_list:
            feature_array = [
                features.Pregnancies,
                features.Glucose,
                features.BloodPressure,
                features.SkinThickness,
                features.Insulin,
                features.BMI,
                features.DiabetesPedigreeFunction,
                features.Age
            ]
            feature_arrays.append(feature_array)
        
        feature_matrix = np.array(feature_arrays)
        
        # Scale features
        scaled_features = scaler.transform(feature_matrix)
        
        # Make predictions
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "prediction": int(pred),
                "probability": float(prob[1]),
                "message": "Diabetic" if pred == 1 else "Not Diabetic",
                "input_id": i
            })
        
        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )