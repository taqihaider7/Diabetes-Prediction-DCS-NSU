from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import logging
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_FILENAME = 'models/best_model.pkl'
SCALER_FILENAME = 'models/scaler.pkl'

# Pydantic models for request/response validation
class DiabetesFeatures(BaseModel):
    Pregnancies: int = Field(ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(ge=0, le=200, description="Glucose level")
    BloodPressure: float = Field(ge=0, le=150, description="Blood pressure")
    SkinThickness: float = Field(ge=0, le=100, description="Skin thickness")
    Insulin: float = Field(ge=0, le=850, description="Insulin level")
    BMI: float = Field(ge=0, le=70, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(ge=0.0, le=3.0, description="Diabetes pedigree function")
    Age: int = Field(ge=1, le=120, description="Age in years")
    
    @validator('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI')
    def validate_positive_values(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

class PredictionResponse(BaseModel):
    prediction_class: int
    probability_no_diabetes: float
    probability_diabetes: float
    model_used: str
    prediction_time: datetime

class BatchPredictionRequest(BaseModel):
    patients: List[DiabetesFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: datetime

# Global variables for model and scaler
model = None
scaler = None
model_type = "Unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, scaler, model_type
    
    logger.info("Loading model and scaler...")
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
            model_type = model_data.get('model_name', type(model).__name__)
        
        logger.info(f"Model and Scaler loaded successfully. Model type: {model_type}")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        # Don't exit - let the app start but return errors for prediction endpoints
        model = None
        scaler = None
    
    yield  # This is where the application runs
    
    # Shutdown
    logger.info("Shutting down...")
    # Cleanup resources if needed

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Diabetes Prediction API",
    description="A FastAPI service for predicting diabetes based on health metrics",
    version="1.0.0",
    lifespan=lifespan
)

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
    
    return response

# --- Health Check Endpoint ---
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=model_type if model is not None else None,
        timestamp=datetime.now()
    )

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: DiabetesFeatures):
    """
    Predict diabetes probability for a single patient
    
    - **Pregnancies**: Number of pregnancies (0-20)
    - **Glucose**: Glucose level (0-200)
    - **BloodPressure**: Blood pressure (0-150)  
    - **SkinThickness**: Skin thickness (0-100)
    - **Insulin**: Insulin level (0-850)
    - **BMI**: Body Mass Index (0-70)
    - **DiabetesPedigreeFunction**: Diabetes pedigree function (0.0-3.0)
    - **Age**: Age in years (1-120)
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert input to DataFrame
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        input_data = np.array([[
            features.Pregnancies,
            features.Glucose,
            features.BloodPressure,
            features.SkinThickness,
            features.Insulin,
            features.BMI,
            features.DiabetesPedigreeFunction,
            features.Age
        ]])
        
        # Scale the data
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict_proba(scaled_input)[0]
        prediction_class = int(model.predict(scaled_input)[0])
        
        logger.info(f"Prediction made - Class: {prediction_class}, Probability: {prediction_prob[1]:.3f}")
        
        return PredictionResponse(
            prediction_class=prediction_class,
            probability_no_diabetes=float(prediction_prob[0]),
            probability_diabetes=float(prediction_prob[1]),
            model_used=model_type,
            prediction_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# --- Batch Prediction Endpoint ---
@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Predict diabetes probability for multiple patients in batch
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
    
    try:
        if len(batch_request.patients) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 patients.")
        
        # Prepare batch data
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        batch_data = []
        for patient in batch_request.patients:
            batch_data.append([
                patient.Pregnancies,
                patient.Glucose,
                patient.BloodPressure,
                patient.SkinThickness,
                patient.Insulin,
                patient.BMI,
                patient.DiabetesPedigreeFunction,
                patient.Age
            ])
        
        input_data = np.array(batch_data)
        
        # Scale the data
        scaled_input = scaler.transform(input_data)
        
        # Make predictions
        prediction_probs = model.predict_proba(scaled_input)
        prediction_classes = model.predict(scaled_input)
        
        # Prepare responses
        predictions = []
        for i, (prob, cls) in enumerate(zip(prediction_probs, prediction_classes)):
            predictions.append(PredictionResponse(
                prediction_class=int(cls),
                probability_no_diabetes=float(prob[0]),
                probability_diabetes=float(prob[1]),
                model_used=model_type,
                prediction_time=datetime.now()
            ))
        
        logger.info(f"Batch prediction completed for {len(predictions)} patients")
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# --- Model Info Endpoint ---
@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_name": model_type,
        "features_used": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "model_loaded": True,
        "timestamp": datetime.now()
    }

# --- Example Data Endpoint ---
@app.get("/example")
async def get_example():
    """Get example input data structure"""
    return {
        "example_input": {
            "Pregnancies": 2,
            "Glucose": 120.0,
            "BloodPressure": 70.0,
            "SkinThickness": 30.0,
            "Insulin": 80.0,
            "BMI": 25.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 35
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )