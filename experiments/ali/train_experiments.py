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
mlflow.set_experiment("Ali_Diabetes_Experiments")


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
        mlflow.set_tag("run_by", "Ali Rehan")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # ---- generic params ----
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_samples", len(train_df) + len(test_df))
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("trained_on", "ali_data_processed")

        # extra model-specific params
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

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
log_reg = LogisticRegression(max_iter=300, solver="lbfgs")

run_and_log(
    log_reg,
    "logistic_regression",
    params={
        "param_max_iter": 300,
        "param_solver": "lbfgs"
    }
)

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    max_features="sqrt"
)

run_and_log(
    rf,
    "random_forest",
    params={
        "param_n_estimators": 200,
        "param_max_depth": "None",
        "param_max_features": "sqrt",
        "param_random_state": 42
    }
)

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=5, weights="uniform")

run_and_log(
    knn,
    "knn",
    params={
        "param_n_neighbors": 5,
        "param_weights": "uniform"
    }
)

# 4. Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

run_and_log(
    gb,
    "gradient_boosting",
    params={
        "param_n_estimators": 100,
        "param_learning_rate": 0.1,
        "param_max_depth": 3,
        "param_random_state": 42
    }
)

