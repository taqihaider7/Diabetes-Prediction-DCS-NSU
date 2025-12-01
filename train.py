import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib


def load_data(csv_path: str = "data/diabetes.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def train_and_log_model():
    # Make sure experiment exists
    mlflow.set_experiment("diabetes-prediction")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameters (could be passed via CLI/MLflow params later)
    n_estimators = 200
    max_depth = 5
    random_state = 42

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Log model in MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save to local file for FastAPI
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "diabetes_rf.pkl"
        joblib.dump(model, model_path)

        # Log the artifact file as well (optional but nice)
        mlflow.log_artifact(str(model_path), artifact_path="models")

        print(f"Model saved to {model_path}")
        print(f"Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    train_and_log_model()
