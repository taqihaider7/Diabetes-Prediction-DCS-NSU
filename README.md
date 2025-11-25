# 🏥 Diabetes Prediction System - DCS NSU

> An end-to-end machine learning project for diabetes prediction using XGBoost with interactive web interfaces, REST APIs, and comprehensive model monitoring.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009485)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Streamlit Web Interface](#streamlit-web-interface)
  - [FastAPI REST API](#fastapi-rest-api)
  - [API Examples](#api-examples)
- [Model Information](#model-information)
- [Data](#data)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a complete machine learning pipeline for diabetes prediction. It combines data science best practices with modern software engineering to create a production-ready prediction system. The system uses XGBoost, one of the most powerful gradient boosting frameworks, trained on the Pima Indians Diabetes Dataset.

### Key Objectives

- **Accurate Predictions**: Binary classification to identify diabetes risk
- **Production-Ready**: Enterprise-level API with proper error handling
- **User-Friendly**: Multiple interfaces (Web UI, REST API, CLI)
- **Explainable**: Transparent predictions with confidence scores
- **Scalable**: Batch processing capabilities for multiple predictions

---

## ✨ Features

### 🎨 User Interfaces

- **Streamlit Web Application**: Interactive dashboard for single and batch predictions
- **FastAPI REST API**: Production-grade API with comprehensive documentation
- **Swagger UI & ReDoc**: Auto-generated interactive API documentation
- **Beautiful Visualizations**: Interactive charts and feature analysis

### 🤖 Machine Learning

- **XGBoost Model**: Optimized gradient boosting classifier
- **Feature Engineering**: 24 engineered features from raw medical data
- **Batch Predictions**: Process multiple patients simultaneously
- **MLflow Integration**: Experiment tracking and model versioning
- **DagsHub Integration**: Remote model storage and versioning

### 📊 Model Management

- **Model Metadata**: Automatic tracking of training parameters
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Information**: Complete feature documentation
- **Health Checks**: Real-time model and API status monitoring

### 📁 Data Management

- **Data Versioning**: Using DVC (Data Version Control)
- **Processed Data**: Normalized and engineered features
- **Train/Test Split**: Proper data handling for model evaluation

---

## 📂 Project Structure

```
Diabetes-Prediction-DCS-NSU/
├── src/                              # Application source code
│   ├── app.py                        # Streamlit web application
│   ├── main.py                       # FastAPI REST API
│   ├── test_api.py                   # API testing script
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # API documentation
│
├── notebooks/                        # Jupyter notebooks
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   ├── model_training_mlflow.ipynb  # Model training & tracking
│   └── artifacts/                    # Generated artifacts
│
├── data/                             # Dataset files
│   ├── diabetes.csv                 # Original raw data
│   └── processed_data.csv           # Processed features
│
├── models/                           # Trained models
│   ├── xgboost_best_model_*.joblib # Serialized model
│   ├── xgboost_best_model_*_metadata.json  # Model metadata
│   └── xgboost_best_model_*.pkl.dvc # DVC tracked model
│
├── data.dvc                          # DVC data versioning
├── src.dvc                           # DVC source tracking
├── LICENSE                           # Apache 2.0 License
└── README.md                         # This file
```

---

## 🛠️ Tech Stack

### Backend & API

- **FastAPI** (v0.104+): Modern async web framework
- **Uvicorn** (v0.24.0): ASGI server
- **Pydantic** (v2.0+): Data validation

### Machine Learning

- **XGBoost**: Gradient boosting classifier
- **scikit-learn**: ML utilities and preprocessing
- **LightGBM**: Alternative gradient boosting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Data & Experiment Tracking

- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **DagsHub**: Remote ML model storage

### Frontend & Visualization

- **Streamlit**: Interactive web UI
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plotting

### Development

- **Python** (3.8+): Programming language
- **Jupyter**: Interactive notebooks
- **Git**: Version control

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)
- ~500MB disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/taqihaider7/Diabetes-Prediction-DCS-NSU.git
cd Diabetes-Prediction-DCS-NSU
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import fastapi, streamlit, xgboost; print('✓ All dependencies installed')"
```

---

## 🚀 Quick Start

### Option 1: Streamlit Web Interface (Recommended for Users)

```bash
cd src
streamlit run app.py
```

Opens automatically at: `http://localhost:8501`

### Option 2: FastAPI REST API (For Developers)

```bash
cd src
python main.py
# or
uvicorn main:app --reload
```

API available at: `http://localhost:8000`

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option 3: Using Python Directly

```bash
cd src
python test_api.py
```

---

## 📖 Usage

### Streamlit Web Application

#### 🔮 Single Patient Prediction

1. Navigate to "Single Prediction" page
2. Fill in patient medical details:
   - Medical History (Pregnancies, Age, Pedigree Function)
   - Physical Measurements (BMI, Blood Pressure, Glucose, etc.)
   - Engineered Features (Auto-filled or custom)
3. Click "Make Prediction"
4. View results with confidence scores and risk level

**Input Fields:**

- **Medical History Tab**
  - Pregnancies: Number of times pregnant (0-20)
  - Age: Patient age in years (1-150)
  - Diabetes Pedigree Function: Family history score (0-2.5)

- **Measurements Tab**
  - Glucose: Blood glucose level (0-500 mg/dL)
  - Blood Pressure: Diastolic BP (0-200 mm Hg)
  - Skin Thickness: Triceps fold thickness (0-100 mm)
  - Insulin: 2-hour serum insulin (0-900 mu U/ml)
  - BMI: Body Mass Index (10-60 kg/m²)

#### 📊 Batch Predictions

1. Go to "Batch Prediction" page
2. Upload a CSV file with multiple patient records
3. Required columns: All 24 feature columns (see feature list below)
4. System validates and processes all records
5. Download results as CSV

**CSV Format Example:**

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,N0,N1,N2,...
6,148,72,35,0,33.6,0.627,50,0.44,0.0,0.0,...
1,85,66,29,0,26.6,0.351,31,0.5,0.1,0.2,...
```

#### ℹ️ Model Information

- View trained model details and metadata
- See performance metrics (Accuracy, ROC-AUC, F1-Score, etc.)
- List all 24 features and their descriptions
- View training date and model source

#### 📈 Data Analysis

- Statistical summaries of all features
- Interactive histograms and box plots
- Correlation matrix heatmap
- Explore data distributions

### FastAPI REST API

#### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoost Classifier",
  "features_count": 24,
  "timestamp": "2024-11-19T10:30:00"
}
```

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
    "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
    "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 0.03,
    "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
    "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
    "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.78,
  "confidence": 0.78,
  "timestamp": "2024-11-19T10:30:00",
  "model_info": {
    "model_name": "XGBoost Classifier",
    "source": "Local File",
    "features_count": 24,
    "training_date": "2024-10-27",
    "best_roc_auc": 0.88,
    "best_accuracy": 0.82
  }
}
```

#### Batch Predictions

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Model Information

```bash
curl -X GET "http://localhost:8000/model_info"
```

#### Get Features

```bash
curl -X GET "http://localhost:8000/features"
```

### API Examples

#### Python with Requests

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Single prediction
patient_data = {
    "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 0.0, "N5": 0.0,
    "N6": 0.0, "N7": 0.0, "N9": 1.0, "N10": 1.0, "N11": 0.0,
    "N15": 0.0, "Pregnancies": 0.64, "Glucose": 0.86, "BloodPressure": 0.03,
    "SkinThickness": 0.67, "Insulin": 0.31, "BMI": 0.17,
    "DiabetesPedigreeFunction": 0.47, "Age": 1.43, "N0": 0.44,
    "N8": 0.14, "N13": 0.56, "N12": 1.20, "N14": 0.02
}

response = requests.post(f"{BASE_URL}/predict", json=patient_data)
result = response.json()

print(f"Prediction: {'Diabetes' if result['prediction'] == 1 else 'No Diabetes'}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🤖 Model Information

### Model Details

- **Algorithm**: XGBoost (eXtreme Gradient Boosting)
- **Task**: Binary Classification
- **Training Date**: October 27, 2024
- **Framework**: scikit-learn compatible

### Performance Metrics

- **Accuracy**: ~82%
- **ROC-AUC**: ~0.88
- **Precision**: High specificity for positive cases
- **Recall**: Good sensitivity for identifying diabetes
- **F1-Score**: Balanced performance metric

### Features (24 Total)

#### Original Features (8)

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-20 |
| Glucose | Plasma glucose level (mg/dL) | 0-200 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 0-120 |
| SkinThickness | Triceps skin fold thickness (mm) | 0-100 |
| Insulin | 2-hour serum insulin (mu U/ml) | 0-900 |
| BMI | Body Mass Index (kg/m²) | 10-65 |
| DiabetesPedigreeFunction | Diabetes family history score | 0.0-2.5 |
| Age | Age in years | 20-80 |

#### Engineered Features (16)

- **N0 to N15**: Automatically engineered features from original medical measurements
- These features capture non-linear relationships and interactions
- Improves model generalization and prediction accuracy

### Training Data

- **Dataset**: Pima Indians Diabetes Dataset
- **Total Samples**: ~768 records
- **Target Variable**: Diabetes (Binary: 0 or 1)
- **Class Distribution**: Imbalanced (positive class ~35%)
- **Data Preprocessing**: Standardization, Missing Value Handling

---

## 📊 Data

### Data Files

#### Raw Data

- **File**: `data/diabetes.csv`
- **Size**: ~26KB
- **Format**: CSV with 9 columns
- **Records**: 768 patients

#### Processed Data

- **File**: `data/processed_data.csv`
- **Size**: ~45KB
- **Features**: 24 engineered features
- **Ready for ML**: Normalized and feature-engineered

### Data Versioning

- Uses **DVC** (Data Version Control)
- Files tracked: `data.dvc`, `src.dvc`
- Enables reproducible ML pipeline
- Remote storage on DagsHub

---

## 📚 Notebooks

### 1. Exploratory Data Analysis (EDA.ipynb)

**Purpose**: Understand data characteristics and patterns

- Dataset overview and statistics
- Missing value analysis
- Feature distributions
- Correlation analysis
- Outlier detection

### 2. Model Training with MLflow (model_training_mlflow.ipynb)

**Purpose**: Train and evaluate models with experiment tracking

- Data preprocessing pipeline
- Model training with XGBoost
- Hyperparameter tuning
- Cross-validation evaluation
- MLflow experiment logging
- Model serialization and export
- Performance metrics visualization

### Running Notebooks

```bash
jupyter notebook notebooks/
```

Or use VS Code with Jupyter extension.

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in `src/` directory (optional):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MLflow Configuration
MLFLOW_TRACKING_URI=file:///mlruns
EXPERIMENT_NAME=Diabetes-Prediction-Experiments

# Model Configuration
MODEL_PATH=../models
MAX_BATCH_SIZE=1000
```

### Streamlit Configuration

Streamlit settings can be customized in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
headless = true
```

---

## 🌐 Deployment

### Local Development

```bash
cd src
streamlit run app.py  # For UI
# OR in another terminal
python main.py        # For API
```

### Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["python", "src/main.py"]
```

Build and run:

```bash
docker build -t diabetes-prediction .
docker run -p 8000:8000 -p 8501:8501 diabetes-prediction
```

### Production Considerations

1. **API Server**: Use `uvicorn` with multiple workers

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Load Balancing**: Deploy behind Nginx or equivalent

3. **Monitoring**: Implement logging and health checks

4. **Security**:
   - Add authentication (JWT tokens)
   - Rate limiting
   - Input validation (already implemented)

5. **Scaling**: Use containerization (Docker, Kubernetes)

---

## 🐛 Troubleshooting

### Common Issues

#### Model Not Loading

**Problem**: `Model not available` error

**Solutions**:

1. Check `models/` directory exists with `.joblib` file
2. Ensure correct Python/library versions
3. Check file permissions
4. Verify path is correct

```bash
ls -la models/  # Check model files exist
```

#### Port Already in Use

**Problem**: `Address already in use` error

**Solutions**:

```bash
# Use different port
streamlit run app.py --server.port 8502
# OR
python main.py --port 8001
```

#### Dependency Issues

**Problem**: `ModuleNotFoundError` or version conflicts

**Solutions**:

```bash
# Upgrade pip first
pip install --upgrade pip

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Check versions
pip list | grep -E "fastapi|streamlit|xgboost"
```

#### Predictions Inconsistent

**Problem**: Different results for same input

**Possible Causes**:

- Model loading from different sources
- Feature ordering incorrect
- Different feature scaling
- XGBoost version differences

**Solutions**:

- Ensure consistent feature order
- Check model metadata
- Verify data preprocessing

#### API Connection Refused

**Problem**: `ConnectionRefusedError` when calling API

**Solutions**:

1. Verify API is running: `python main.py`
2. Check correct URL: `http://localhost:8000`
3. Check firewall settings
4. Use `-v` flag for verbose output

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** changes: `git commit -m "Add your feature"`
4. **Push** to branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) file for details.

**You are free to:**

- Use commercially
- Modify the code
- Distribute
- Use privately

**Conditions:**

- Include original copyright notice
- Include license copy
- Include changelog

---

## 📞 Support & Contact

For issues, questions, or suggestions:

1. **GitHub Issues**: [Create an issue](https://github.com/taqihaider7/Diabetes-Prediction-DCS-NSU/issues)
2. **Documentation**: Check README files in each directory
3. **Code Examples**: See `notebooks/` and `src/test_api.py`

---

## 🔗 Related Resources

- **Dataset**: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- **XGBoost Docs**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io/)
- **DVC**: [dvc.org](https://dvc.org/)
- **MLflow**: [mlflow.org](https://mlflow.org/)

---

## 🎓 Project Context

**Course**: Distributed Computing System (DCS)  
**Semester**: 3rd Semester  
**University**: Novosibirsk State University, Russia (NSU)  
**Project Type**: Machine Learning Operations Capstone Project  

---

## 📊 Project Stats

- **Code Files**: 4 main Python modules
- **Notebooks**: 2 comprehensive Jupyter notebooks
- **Features**: 24 engineered features
- **Model Accuracy**: ~82%
- **API Endpoints**: 6 main endpoints
- **Lines of Code**: ~2000+

---

## ✅ Checklist for Users

- [ ] Clone the repository
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -r src/requirements.txt`)
- [ ] Run Streamlit: `streamlit run src/app.py`
- [ ] Or run API: `python src/main.py`
- [ ] Visit <http://localhost:8501> or <http://localhost:8000>
- [ ] Make test predictions
- [ ] Explore model information

---

## 🎉 Acknowledgments

- **Dataset Source**: UCI Machine Learning Repository
- **Libraries**: XGBoost, FastAPI, Streamlit, Scikit-learn teams
- **Frameworks**: MLflow, DVC communities
- **Institution**: North South University

---

**Last Updated**: November 19, 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅
