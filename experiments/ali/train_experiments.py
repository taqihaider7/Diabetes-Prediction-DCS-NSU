import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

import joblib


# ========= 1. Connect MLflow to DagsHub =========

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri is None:
    raise ValueError("MLFLOW_TRACKING_URI is not set. Please export it in your terminal.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("ali_diabetes_experiments")


# ========= 2. Load YOUR processed data (ali_data) =========

# Base directory: repo root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

train_path = os.path.join(BASE_DIR, "ali_data", "processed", "train_processed.csv")
test_path  = os.path.join(BASE_DIR, "ali_data", "processed", "test_processed.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop("Outcome", axis=1)
y_train = train_df["Outcome"]

X_test = test_df.drop("Outcome", axis=1)
y_test = test_df["Outcome"]


# ========= 3. Helper to train & log each model =========

def run_and_log(model, model_name, params=None):
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log params
        mlflow.log_param("model_type", model_name)
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # ---- Model logging (DagsHub-compatible) ----
        local_model_path = f"{model_name}_model.pkl"
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(local_model_path)
        # -------------------------------------------

        print(f"{model_name} -> accuracy={acc:.4f}, f1={f1:.4f}")


# ========= 4. Run 4 Different Models =========

# 1. Logistic Regression
run_and_log(
    LogisticRegression(max_iter=300),
    "logistic_regression",
    params={"max_iter": 300}
)

# 2. Random Forest
run_and_log(
    RandomForestClassifier(n_estimators=200, random_state=42),
    "random_forest",
    params={"n_estimators": 200}
)

# 3. KNN
run_and_log(
    KNeighborsClassifier(n_neighbors=5),
    "knn",
    params={"n_neighbors": 5}
)

# 4. Gradient Boosting
run_and_log(
    GradientBoostingClassifier(),
    "gradient_boosting",
    params={}
)
