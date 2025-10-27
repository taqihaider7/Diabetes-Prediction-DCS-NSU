# Diabetes Prediction FastAPI Application

A production-ready FastAPI application for making diabetes predictions using XGBoost and LightGBM models trained with MLflow and logged to DagsHub.

## 🎯 Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Lifespan Context Manager**: Proper startup and shutdown management
- **Dual Model Support**: XGBoost and LightGBM models
- **MLflow Integration**: Automatically loads models from DagsHub
- **Batch Processing**: Support for single and batch predictions
- **Health Checks**: Application and model status monitoring
- **CORS Enabled**: Cross-origin requests allowed
- **Comprehensive Logging**: Detailed logging for debugging
- **Auto Documentation**: Interactive API docs with Swagger UI and ReDoc
- **Error Handling**: Robust exception handling with meaningful error messages

## 📋 Requirements

Python 3.8+

See `requirements.txt` for all dependencies:

- fastapi
- uvicorn
- pydantic
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- joblib
- mlflow

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MLflow Access (Optional)

The application automatically loads models from DagsHub. If you want to use a specific run ID:

Edit the `MLFLOW_TRACKING_URI` in `main.py`:

```python
MLFLOW_TRACKING_URI = "https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow"
```

### 3. Model Loading

The application tries to load models in this order:

1. **MLflow/DagsHub**: Downloads from the best model run
2. **Local Fallback**: Looks for `.joblib` files in `../models/` directory

## ▶️ Running the Application

### Development Mode (with auto-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python directly

```bash
python main.py
```

The application will start at: `http://localhost:8000`

## 📚 API Documentation

Once running, access the interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🔌 API Endpoints

### 1. Health Check

**GET** `/health`

Check application and model loading status.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": {
    "xgboost": true,
    "lightgbm": true
  },
  "timestamp": "2024-10-27T12:00:00.000000",
  "uptime_info": {
    "models_available": "2",
    "total_models": "2"
  }
}
```

### 2. Model Information

**GET** `/model-info`

Get available models and required features.

**Response:**

```json
{
  "available_models": ["xgboost", "lightgbm"],
  "model_status": {
    "xgboost": "loaded",
    "lightgbm": "loaded"
  },
  "experiment_name": "Diabetes-Prediction-Experiments-vscode",
  "tracking_uri": "https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow",
  "features": [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
  ],
  "feature_count": 8
}
```

### 3. Single Prediction (HTTP Request Method #1)

**POST** `/predict`

Make a single diabetes prediction.

**Request Body:**

```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Query Parameters:**

- `model_type` (string, default: "xgboost"): Model to use ("xgboost" or "lightgbm")

**Response:**

```json
{
  "prediction": 1,
  "probability": {
    "no_diabetes": 0.25,
    "diabetes": 0.75
  },
  "confidence": 0.75,
  "model_used": "xgboost",
  "timestamp": "2024-10-27T12:00:00.000000"
}
```

### 4. Batch Prediction (HTTP Request Method #2)

**POST** `/predict-batch`

Make predictions for multiple patients.

**Request Body:**

```json
{
  "data": [
    {
      "Pregnancies": 6,
      "Glucose": 148,
      "BloodPressure": 72,
      "SkinThickness": 35,
      "Insulin": 0,
      "BMI": 33.6,
      "DiabetesPedigreeFunction": 0.627,
      "Age": 50
    },
    {
      "Pregnancies": 1,
      "Glucose": 85,
      "BloodPressure": 66,
      "SkinThickness": 29,
      "Insulin": 0,
      "BMI": 26.6,
      "DiabetesPedigreeFunction": 0.351,
      "Age": 31
    }
  ],
  "model_type": "xgboost"
}
```

**Response:**

```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": {
        "no_diabetes": 0.25,
        "diabetes": 0.75
      },
      "confidence": 0.75,
      "model_used": "xgboost",
      "timestamp": "2024-10-27T12:00:00.000000"
    },
    {
      "prediction": 0,
      "probability": {
        "no_diabetes": 0.85,
        "diabetes": 0.15
      },
      "confidence": 0.85,
      "model_used": "xgboost",
      "timestamp": "2024-10-27T12:00:00.000000"
    }
  ],
  "total_records": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "processing_time_seconds": 0.0234
}
```

## 💡 Usage Examples

### Using cURL

**Single Prediction:**

```bash
curl -X POST "http://localhost:8000/predict?model_type=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

**Health Check:**

```bash
curl "http://localhost:8000/health"
```

### Using Python Requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    params={"model_type": "xgboost"},
    json={
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
)

print(response.json())

# Batch prediction
batch_data = {
    "data": [
        {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    ],
    "model_type": "xgboost"
}

response = requests.post(
    "http://localhost:8000/predict-batch",
    json=batch_data
)

print(response.json())
```

### Using Postman

1. Open Postman
2. Create a new POST request
3. URL: `http://localhost:8000/predict`
4. Add query parameter: `model_type=xgboost`
5. Body (JSON):

```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

## 🔄 Lifespan Management

The application uses FastAPI's lifespan context manager for proper startup and shutdown:

### Startup (lifespan entry)

- MLflow tracking URI is configured
- Models are loaded from DagsHub or local storage
- Application status is logged
- Health checks are enabled

### Shutdown (lifespan exit)

- Graceful shutdown is performed
- Uptime statistics are logged
- Resources are cleaned up

## 📊 Logging

All operations are logged with detailed information:

- Model loading status
- Prediction requests (input and output)
- Performance metrics (processing time)
- Errors and exceptions
- Application startup/shutdown

Check the console output or logs for detailed information.

## 🛡️ Error Handling

The application includes comprehensive error handling:

- **400 Bad Request**: Invalid model type or missing features
- **500 Internal Server Error**: Model loading failures or prediction errors
- **503 Service Unavailable**: Models not available

Error responses include:

- Error message
- Timestamp
- Request path

Example error response:

```json
{
  "error": "Invalid model type. Available: ['xgboost', 'lightgbm']",
  "timestamp": "2024-10-27T12:00:00.000000",
  "path": "http://localhost:8000/predict?model_type=invalid"
}
```

## 📁 Project Structure

```
src/
├── main.py                  # Main FastAPI application
└── requirements.txt         # Python dependencies
```

## 🔗 Model Source

Models are automatically downloaded from:
**DagsHub Repository**: <https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU>

## 🤝 Contributing

To extend the application:

1. Add new endpoints by defining routes with `@app.get()` or `@app.post()`
2. Create new Pydantic models for request/response validation
3. Add logging for new features
4. Update documentation

## 📝 Notes

- The application requires internet access to download models from DagsHub on first run
- Models are cached locally after first download
- All timestamps are in ISO format
- Predictions return probabilities for both classes (0 and 1)

## 🐛 Troubleshooting

### Models not loading from MLflow

1. Check internet connection
2. Verify DagsHub repository access
3. Check logs for specific error messages
4. Ensure models exist in `../models/` directory as fallback

### Port already in use

Use a different port:

```bash
uvicorn main:app --port 8001
```

### Missing dependencies

Install them:

```bash
pip install -r requirements.txt --upgrade
```

## 📞 Support

For issues or questions:

1. Check the detailed logs in console output
2. Review API documentation at `/docs`
3. Check the GitHub repository
4. Review the model training notebook for model details

---

**Version**: 1.0.0  
**Last Updated**: October 27, 2024
