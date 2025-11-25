#   📘 Diabetes Prediction – Individual MLOps Project (Ali)
#   Branch: ds-ali | Student: Ali Rehan | NSU – MLOps Course

title: "**Diabetes Prediction – Individual MLOps Experiments**"
author: "**Ali Rehan**"
branch: "**ds-ali**"
course: "**Distributed Computing Systems (MLOps)** – North South University"
dataset: "**Pima Indians Diabetes Dataset** (Kaggle)"
overview: |
  This branch (`ds-ali`) contains **my individual work** for the final MLOps project.
  The branch includes **EDA**, **preprocessing**, **ML training**, **MLflow experiment logging**, and **best model selection**.
  All work is done according to the professor's requirements where each student must develop their own experiment pipeline.

# 📁 Project Structure (Ali Only)

structure:
  ali_data:
    raw:
      - **diabetes.csv**
    processed:
      - **train_processed.csv**
      - **test_processed.csv**
  notebooks:
    ali:
      - **ali_eda.ipynb**
      - **ali_preprocessing.ipynb**
  experiments:
    ali:
      - **train_experiments.py**
  config:
    - **ali_best_model.json**
  additional:
    - **requirements.txt**
    - **README.md**

# 📝 EDA – Exploratory Data Analysis

eda:
  file: "**notebooks/ali/ali_eda.ipynb**"
  highlights: |
    - Performed complete statistical analysis on the diabetes dataset.
    - Checked for missing/zero values and treated them where needed.
    - Visualized distributions using histograms and boxplots.
    - Studied correlation matrix to understand feature relationships.
    - Extracted insights on which features impact diabetes outcome.

# 🧹 Preprocessing

preprocessing:
  file: "**notebooks/ali/ali_preprocessing.ipynb**"
  tasks: |
    - Cleaned zero-value entries in medically relevant columns.
    - Applied feature scaling.
    - Split dataset into **80/20 train-test**.
    - Stored processed datasets in:
      - `ali_data/processed/train_processed.csv`
      - `ali_data/processed/test_processed.csv`

# 🤖 Model Training & Experiments (MLflow + DagsHub)

experiments:
  file: "**experiments/ali/train_experiments.py**"
  workflow: |
    - Connected MLflow to **DagsHub tracking server**.
    - Loaded *my own* processed data (not shared group data).
    - Trained **4 ML models** with logged parameters and metrics:
        1. **Logistic Regression**
        2. **Random Forest**
        3. **K-Nearest Neighbors**
        4. **Gradient Boosting**
    - Logged model artifacts (`.pkl`) and hyperparameters.
    - Added run tags such as:
        - **owner: Ali**
        - **student: Ali Rehan**
        - **branch: ds-ali**
    - All experiments are visible in the DagsHub MLflow UI.

models_trained:
  - "**Logistic Regression** – baseline model"
  - "**Random Forest** – strong tree-based model"
  - "**KNN** – distance-based model"
  - "**Gradient Boosting** – best performing ensemble model"

# 🏆 Best Model Selection

best_model:
  file: "**config/ali_best_model.json**"
  description: |
    This JSON file stores metadata for my best MLflow run:
    - tracking URI
    - experiment name
    - run_id from DagsHub
    - best model artifact name (e.g., gradient_boosting_model.pkl)
    - selection based on **highest accuracy** and **F1 score**
    - owner information (Ali)

# 📦 Dependencies

dependencies:
  install: "**pip install -r requirements.txt**"
  file: "**requirements.txt**"
  includes:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - mlflow
    - joblib

# 🧪 Re-running My Experiments

rerun_instructions:
  activate_env: "source venv/bin/activate"
  set_env_vars: |
    export MLFLOW_TRACKING_URI="https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow"
    export MLFLOW_TRACKING_USERNAME="alijoya512"
    export MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"
  run_script: |
    cd experiments/ali
    python train_experiments.py

# 🎯 Summary of My Work

summary: |
  - Completed full **EDA** and **preprocessing pipeline** for diabetes dataset.
  - Trained **four different machine learning models**.
  - Logged all runs to **DagsHub MLflow** with parameters, metrics, artifacts, and tags.
  - Selected best-performing model and stored metadata in `ali_best_model.json`.
  - Branch `ds-ali` contains all of my individual contributions clearly separated.
  - Fully satisfies requirement: *"Every member must have their own data experiments."*
