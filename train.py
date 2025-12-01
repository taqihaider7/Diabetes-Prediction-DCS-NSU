import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


# ==========================================================
# 1. MLflow Setup (DagsHub or Local)
# ==========================================================

# --- Environment Variables (take from system if available) ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username     = os.getenv("MLFLOW_TRACKING_USERNAME")
password     = os.getenv("MLFLOW_TRACKING_PASSWORD")

# --- Fallback values for local development (safe to use locally) ---
if not tracking_uri:
    tracking_uri = "https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow"

if not username:
    username = "tahanoor78655"

if not password:
    password = "53dfbba532a140056bf61501adaa2da2e1659338"   # Replace later for safety


# Apply authentication for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

# Connect MLflow
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("taha-DiabetesExperiments")

print(f"✔ MLflow connected to: {tracking_uri}")
print(f"✔ Using username: {username}")


# Enable MLflow autologging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)


# ==========================================================
# 2. Load Data
# ==========================================================

# FIXED → project root = folder where train.py is located
BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR  /"data"/ "processed" / "train_processed.csv"
test_path  = BASE_DIR / "data" / "processed" / "test_processed.csv"

# Validate file existence
if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        f"❌ Missing data files:\n{train_path}\n{test_path}\n"
        "➡ Make sure preprocessing has created these files."
    )

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

X_train = train_df.drop(columns=["Outcome"])
y_train = train_df["Outcome"]

X_test = test_df.drop(columns=["Outcome"])
y_test = test_df["Outcome"]


# ==========================================================
# 3. Training Function
# ==========================================================

def run_experiment(model, model_name, params=None):
    """Train model, evaluate, and log results to MLflow."""

    with mlflow.start_run(run_name=model_name):

        # Log params
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save model locally
        model_path = f"{model_name}.pkl"
        joblib.dump(model, model_path)

        # Log model artifact
        mlflow.log_artifact(model_path)

        print(f"✔ {model_name} logged | accuracy={acc:.4f}, f1={f1:.4f}")


# ==========================================================
# 4. Train Models
# ==========================================================

models_to_run = [
    (
        LogisticRegression(max_iter=300),
        "logistic_regression",
        {"max_iter": 300},
    ),
    (
        RandomForestClassifier(n_estimators=200, random_state=42),
        "random_forest",
        {"n_estimators": 200, "random_state": 42},
    ),
    (
        GradientBoostingClassifier(),
        "gradient_boosting",
        {},
    ),
]

for model, name, params in models_to_run:
    run_experiment(model, name, params)
