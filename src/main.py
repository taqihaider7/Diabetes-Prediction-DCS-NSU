"""
FastAPI Application for Diabetes Prediction using XGBoost Model
This application uses the best XGBoost model trained and logged to DagsHub for inference.

Features:
- Lifespan management for model loading/unloading
- Multiple HTTP endpoints for predictions
- Request validation with Pydantic
- Comprehensive error handling
- Health checks and model information
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import sys
import json
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Global Variables ====================
model = None
model_metadata = None
feature_names = None
mlflow_client = None
model_info = {}

# ==================== Feature Configuration ====================
# Features from the processed_data.csv
FEATURE_COLUMNS = [
    'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N9', 'N10', 'N11', 'N15',
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'N0', 'N8', 'N13', 'N12', 'N14'
]

# ==================== Pydantic Models ====================
class DiabetesPredictionInput(BaseModel):
    """Input model for diabetes prediction"""
    N1: float = Field(..., description="Feature N1")
    N2: float = Field(..., description="Feature N2")
    N3: float = Field(..., description="Feature N3")
    N4: float = Field(..., description="Feature N4")
    N5: float = Field(..., description="Feature N5")
    N6: float = Field(..., description="Feature N6")
    N7: float = Field(..., description="Feature N7")
    N9: float = Field(..., description="Feature N9")
    N10: float = Field(..., description="Feature N10")
    N11: float = Field(..., description="Feature N11")
    N15: float = Field(..., description="Feature N15")
    Pregnancies: float = Field(..., description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, description="Glucose level")
    BloodPressure: float = Field(..., ge=0, description="Blood pressure")
    SkinThickness: float = Field(..., ge=0, description="Skin thickness")
    Insulin: float = Field(..., ge=0, description="Insulin level")
    BMI: float = Field(..., gt=0, description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: float = Field(..., gt=0, le=150, description="Age in years")
    N0: float = Field(..., description="Feature N0")
    N8: float = Field(..., description="Feature N8")
    N13: float = Field(..., description="Feature N13")
    N12: float = Field(..., description="Feature N12")
    N14: float = Field(..., description="Feature N14")

    model_config = {
        "json_schema_extra": {
            "example": {
                "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
                "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
                "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 0.03,
                "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
                "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
                "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
            }
        }
    }


class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    predictions: List[DiabetesPredictionInput] = Field(..., description="List of prediction requests")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
                        "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
                        "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 0.03,
                        "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
                        "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
                        "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
                    }
                ]
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for a single prediction"""
    prediction: int = Field(..., description="Predicted class (0: No Diabetes, 1: Diabetes)")
    probability: float = Field(..., ge=0, le=1, description="Probability of positive class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    total_predictions: int = Field(..., description="Total predictions made")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    timestamp: str = Field(..., description="Batch processing timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loading status")
    model_name: Optional[str] = Field(..., description="Name of loaded model")
    features_count: int = Field(..., description="Number of expected features")
    timestamp: str = Field(..., description="Health check timestamp")


# ==================== Model Loading Functions ====================
def load_model_from_local(model_path: str) -> tuple:
    """Load model from local file system"""
    try:
        logger.info(f"Loading model from local path: {model_path}")
        model_obj = joblib.load(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        return model_obj, "Local File"
    except Exception as e:
        logger.error(f"Failed to load model from local path: {str(e)}")
        raise


def load_model_from_dagshub(run_id: str, artifact_path: str = "production_models") -> tuple:
    """Load model from DagsHub using MLflow"""
    try:
        logger.info(f"Loading model from DagsHub (Run ID: {run_id})")
        
        # Download artifact from MLflow
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}",
            tracking_uri=mlflow.get_tracking_uri()
        )
        
        logger.info(f"Downloaded artifacts to: {local_path}")
        
        # Find model file
        model_files = [f for f in os.listdir(local_path) if f.endswith('.joblib')]
        
        if not model_files:
            raise FileNotFoundError(f"No .joblib model file found in {local_path}")
        
        model_file = os.path.join(local_path, model_files[0])
        model_obj = joblib.load(model_file)
        
        logger.info(f"✓ Model loaded successfully from DagsHub (Run: {run_id})")
        return model_obj, f"DagsHub (Run: {run_id})"
    
    except Exception as e:
        logger.error(f"Failed to load model from DagsHub: {str(e)}")
        raise


def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """Load model metadata from metadata.json file"""
    try:
        # Construct metadata path
        base_name = os.path.basename(model_path).replace('.joblib', '').replace('.pkl', '')
        metadata_path = os.path.join(os.path.dirname(model_path), f"{base_name}_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Model metadata loaded from {metadata_path}")
            return metadata
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
            return {}
    
    except Exception as e:
        logger.error(f"Failed to load model metadata: {str(e)}")
        return {}


# ==================== Lifespan Context Manager ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    - Startup: Load model and initialize resources
    - Shutdown: Clean up resources
    """
    # ========== STARTUP ==========
    logger.info("🚀 Starting up Diabetes Prediction API...")
    
    global model, model_metadata, feature_names, mlflow_client, model_info
    
    try:
        # Initialize MLflow
        logger.info("Initializing MLflow client...")
        mlflow_client = MlflowClient()
        mlflow.set_tracking_uri("file:///mlruns")
        logger.info(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Try to load model from DagsHub first, fallback to local
        models_dir = os.path.join(os.getcwd(), "..", "models")
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            if model_files:
                # Use the first (most recent) model file
                model_path = os.path.join(models_dir, model_files[0])
                logger.info(f"Found model file: {model_files[0]}")
                
                try:
                    model, source = load_model_from_local(model_path)
                    model_metadata = load_model_metadata(model_path)
                except Exception as e:
                    logger.error(f"Failed to load from local: {str(e)}")
                    raise
            else:
                raise FileNotFoundError("No .joblib model files found in models directory")
        else:
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Setup model info
        model_info = {
            "model_name": "XGBoost Classifier",
            "source": source,
            "features_count": len(FEATURE_COLUMNS),
            "training_date": model_metadata.get('training_date', 'Unknown'),
            "model_type": model_metadata.get('model_type', 'XGBoost'),
            "best_roc_auc": model_metadata.get('full_data_metrics', {}).get('roc_auc', 'Unknown'),
            "best_accuracy": model_metadata.get('full_data_metrics', {}).get('accuracy', 'Unknown'),
        }
        
        feature_names = FEATURE_COLUMNS
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Model Type: {model_info['model_type']}")
        logger.info(f"   Features: {len(FEATURE_COLUMNS)}")
        logger.info(f"   Source: {source}")
        logger.info(f"   ROC-AUC: {model_info['best_roc_auc']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {str(e)}")
        model = None
        model_metadata = {}
        raise
    
    yield
    
    # ========== SHUTDOWN ==========
    logger.info("🛑 Shutting down Diabetes Prediction API...")
    logger.info("Cleaning up resources...")
    
    # Clean up
    model = None
    model_metadata = None
    feature_names = None
    mlflow_client = None
    
    logger.info("✓ Cleanup completed")
    logger.info("👋 API shutdown successful")


# ==================== FastAPI Application ====================
app = FastAPI(
    title="Diabetes Prediction API",
    description="ML-powered API for diabetes prediction using XGBoost model",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== HTTP Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - Welcome message"""
    return {
        "message": "Welcome to Diabetes Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns the status of the API and model availability
    """
    logger.info("Health check requested")
    
    return HealthCheckResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model_info.get("model_name"),
        features_count=len(FEATURE_COLUMNS),
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: DiabetesPredictionInput):
    """
    Single prediction endpoint
    
    Accepts patient medical features and returns diabetes prediction
    with probability and confidence scores
    
    **Returns:**
    - prediction: 0 (No Diabetes) or 1 (Diabetes)
    - probability: Probability of positive class
    - confidence: Model confidence in prediction
    - timestamp: When prediction was made
    - model_info: Metadata about the model used
    """
    
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        logger.info("Processing single prediction request")
        
        # Convert input to DataFrame with correct feature order
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
        
        logger.info(f"Input shape: {input_df.shape}")
        logger.info(f"Features: {list(input_df.columns)}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Calculate confidence
        confidence = float(np.max(prediction_proba))
        probability = float(prediction_proba[1])
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Confidence: {confidence:.4f}")
        
        response = PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            model_info=model_info
        )
        
        logger.info("✓ Prediction completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(batch_input: BatchPredictionInput, background_tasks: BackgroundTasks):
    """
    Batch prediction endpoint
    
    Accepts multiple patient records and returns predictions for all
    Useful for processing multiple patients at once
    
    **Parameters:**
    - predictions: List of prediction requests
    
    **Returns:**
    - total_predictions: Total number of records processed
    - successful_predictions: Number of successful predictions
    - failed_predictions: Number of failed predictions
    - predictions: List of all predictions with scores
    - timestamp: When batch was processed
    """
    
    if model is None:
        logger.error("Model not loaded for batch prediction")
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        logger.info(f"Processing batch prediction with {len(batch_input.predictions)} records")
        
        results = []
        failed_count = 0
        
        for idx, input_data in enumerate(batch_input.predictions):
            try:
                # Convert to DataFrame
                input_dict = input_data.dict()
                input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                confidence = float(np.max(prediction_proba))
                probability = float(prediction_proba[1])
                
                results.append({
                    "record_id": idx + 1,
                    "prediction": int(prediction),
                    "probability": probability,
                    "confidence": confidence,
                    "status": "success"
                })
            
            except Exception as e:
                logger.warning(f"Failed to process record {idx + 1}: {str(e)}")
                results.append({
                    "record_id": idx + 1,
                    "error": str(e),
                    "status": "failed"
                })
                failed_count += 1
        
        successful_count = len(batch_input.predictions) - failed_count
        
        logger.info(f"Batch processing completed: {successful_count}/{len(batch_input.predictions)} successful")
        
        return BatchPredictionResponse(
            total_predictions=len(batch_input.predictions),
            successful_predictions=successful_count,
            failed_predictions=failed_count,
            predictions=results,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model_info", tags=["Model"])
async def get_model_info():
    """
    Get detailed information about the loaded model
    
    **Returns:**
    - Model type and source (local or DagsHub)
    - Feature count and names
    - Training metadata
    - Performance metrics
    """
    
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        logger.info("Retrieving model information")
        
        response = {
            "model_info": model_info,
            "features": {
                "count": len(FEATURE_COLUMNS),
                "names": FEATURE_COLUMNS
            },
            "metadata": model_metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("✓ Model information retrieved successfully")
        return response
    
    except Exception as e:
        logger.error(f"Failed to retrieve model info: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve model info: {str(e)}")


@app.get("/features", tags=["Model"])
async def get_features():
    """
    Get list of required features for predictions
    
    **Returns:**
    - List of all required feature names in order
    - Feature count
    """
    
    logger.info("Features information requested")
    
    return {
        "features": FEATURE_COLUMNS,
        "count": len(FEATURE_COLUMNS),
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ==================== Main ====================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Diabetes Prediction API...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
