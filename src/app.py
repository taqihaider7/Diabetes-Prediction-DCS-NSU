from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import uvicorn

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
        
        # Debug: print what's in the model data
        print(f"📦 Model data keys: {list(model_data.keys())}")
        
        # Handle different model storage formats
        if isinstance(model_data, dict):
            # New format with dictionary
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            model_metadata = model_data.get('metadata', {})
        else:
            # Old format - assume it's the model directly
            model = model_data
            scaler = StandardScaler()  # You'll need to refit this
            model_metadata = {}
        
        if model is None:
            print("❌ Model is None after loading")
            return
            
        print("✅ Model loaded successfully")
        print(f"📊 Model type: {type(model).__name__}")
        if model_metadata:
            print(f"🎯 Model accuracy: {model_metadata.get('accuracy', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
        json_schema_extra = {  # Fixed from schema_extra to json_schema_extra
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

class BatchPredictionRequest(BaseModel):
    instances: List[DiabetesFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/")
async def root():
    return {
        "message": "Diabetes Prediction API",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - Model not loaded. Please train the model first."
        )
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: DiabetesFeatures):
    """Make a single prediction"""
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
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability[1]),
            message="Diabetic" if prediction == 1 else "Not Diabetic",
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(batch_request: BatchPredictionRequest):
    """Make batch predictions"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        predictions = []
        
        for features in batch_request.instances:
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
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                probability=float(probability[1]),
                message="Diabetic" if prediction == 1 else "Not Diabetic",
                confidence=confidence
            ))
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    info = {
        "model_type": type(model).__name__,
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "feature_descriptions": {
            "Pregnancies": "Number of pregnancies",
            "Glucose": "Plasma glucose concentration",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index",
            "DiabetesPedigreeFunction": "Diabetes pedigree function",
            "Age": "Age in years"
        }
    }
    
    if model_metadata:
        info.update(model_metadata)
    
    return info

@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for model updates)"""
    load_model()
    return {"message": "Model reloaded successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Changed to string import for reload
        host="0.0.0.0", 
        port=8000,
        reload=True
    )