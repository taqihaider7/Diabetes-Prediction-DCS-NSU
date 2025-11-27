# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# import joblib
# import mlflow
# import mlflow.sklearn
# import os
# from pathlib import Path
# import sklearn

# class DiabetesModelTrainer:
#     def __init__(self):
#         self.model = None
#         self.scaler = StandardScaler()
#         self.best_score = 0
        
#         # Configuration
#         self.MLFLOW_TRACKING_URI = "mlruns"
#         self.MLFLOW_EXPERIMENT_NAME = "diabetes_prediction"
#         self.MODEL_DIR = "../models"
#         self.BEST_MODEL_NAME = "best_diabetes_model.pkl"
#         self.DATA_PATH = "data/diabetes.csv"
#         self.TEST_SIZE = 0.2
#         self.RANDOM_STATE = 42
        
#         # Create directories if they don't exist
#         os.makedirs(self.MODEL_DIR, exist_ok=True)
#         os.makedirs(self.MLFLOW_TRACKING_URI, exist_ok=True)
    
#     def load_data(self):
#         """Load and preprocess the diabetes dataset"""
#         df = pd.read_csv(self.DATA_PATH)
        
#         # Handle missing values (zeros that should be NaN)
#         columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#         for col in columns_to_clean:
#             df[col] = df[col].replace(0, np.nan)
#             df[col] = df[col].fillna(df[col].median())
        
#         return df
    
#     def prepare_features(self, df):
#         """Prepare features and target"""
#         X = df.drop('Outcome', axis=1)
#         y = df['Outcome']
        
#         return X, y
    
#     def train_model(self):
#         """Train the model with MLflow tracking"""
#         # Set up MLflow
#         mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
#         mlflow.set_experiment(self.MLFLOW_EXPERIMENT_NAME)
        
#         # Load and prepare data
#         df = self.load_data()
#         X, y = self.prepare_features(df)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE, stratify=y
#         )
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         with mlflow.start_run():
#             # Train model
#             self.model = RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=10,
#                 random_state=self.RANDOM_STATE
#             )
            
#             self.model.fit(X_train_scaled, y_train)
            
#             # Make predictions
#             y_pred = self.model.predict(X_test_scaled)
#             y_pred_proba = self.model.predict_proba(X_test_scaled)
            
#             # Calculate metrics
#             accuracy = accuracy_score(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
#             cr = classification_report(y_test, y_pred, output_dict=True)
            
#             # Log parameters
#             mlflow.log_param("n_estimators", 100)
#             mlflow.log_param("max_depth", 10)
#             mlflow.log_param("test_size", self.TEST_SIZE)
#             mlflow.log_param("random_state", self.RANDOM_STATE)
            
#             # Log metrics
#             mlflow.log_metric("accuracy", accuracy)
#             mlflow.log_metric("precision", cr['1']['precision'])
#             mlflow.log_metric("recall", cr['1']['recall'])
#             mlflow.log_metric("f1_score", cr['1']['f1-score'])
            
#             # Log model
#             mlflow.sklearn.log_model(self.model, "random_forest_model")
            
#             # Log artifacts
#             mlflow.log_artifact(self.DATA_PATH)
            
#             print(f"Model trained with accuracy: {accuracy:.4f}")
#             print(f"Precision: {cr['1']['precision']:.4f}")
#             print(f"Recall: {cr['1']['recall']:.4f}")
#             print(f"F1-Score: {cr['1']['f1-score']:.4f}")
            
#             # Save best model locally
#             if accuracy > self.best_score:
#                 self.best_score = accuracy
#                 self.save_model()
                
#             return accuracy

# def save_model(self):
#     """Save the best model locally"""
#     model_path = f"{self.MODEL_DIR}/{self.BEST_MODEL_NAME}"
    
#     # Create a more robust model dictionary
#     model_dict = {
#         'model': self.model,
#         'scaler': self.scaler,
#         'metadata': {
#             'accuracy': self.best_score,
#             'model_type': 'RandomForest',
#             'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
#             'sklearn_version': sklearn.__version__,
#             'numpy_version': np.__version__,
#             'pandas_version': pd.__version__
#         }
#     }
    
#     joblib.dump(model_dict, model_path)
#     print(f"✅ Best model saved to {model_path}")

#     # def save_model(self):
    
#     #     """Save the best model locally"""
#     #     model_path = f"{self.MODEL_DIR}/{self.BEST_MODEL_NAME}"
#     #     joblib.dump({
#     #         'model': self.model,
#     #         'scaler': self.scaler,
#     #         'metadata': {
#     #             'accuracy': self.best_score,
#     #             'model_type': 'RandomForest',
#     #             'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#     #                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#     #         }
#     #     }, model_path)
        
#     #     print(f"Best model saved to {model_path}")

# def register_model_dagshub():
#     """Register the best model for DAGsHub (manual integration)"""
#     mlflow.set_tracking_uri("mlruns")
#     experiment_name = "diabetes_prediction"
    
#     try:
#         # Search for the best run
#         experiment = mlflow.get_experiment_by_name(experiment_name)
#         if experiment is None:
#             print("No experiment found")
#             return
        
#         runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
#         if runs.empty:
#             print("No runs found in experiment")
#             return
            
#         best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        
#         print(f"Best run ID: {best_run.run_id}")
#         print(f"Best accuracy: {best_run['metrics.accuracy']:.4f}")
        
#         # Model URI for registration
#         model_uri = f"runs:/{best_run.run_id}/random_forest_model"
        
#         print("=" * 50)
#         print("DAGsHub Registration Instructions:")
#         print("1. Set up your DAGsHub repository")
#         print("2. Configure MLflow tracking URI:")
#         print("   mlflow.set_tracking_uri('https://dagshub.com/<username>/<repository>.mlflow')")
#         print("3. Use this model URI to register:")
#         print(f"   Model URI: {model_uri}")
#         print("4. Run: mlflow.register_model(model_uri, 'diabetes-prediction-model')")
#         print("=" * 50)
        
#     except Exception as e:
#         print(f"Error in model registration: {e}")

# if __name__ == "__main__":
#     # Train the model
#     trainer = DiabetesModelTrainer()
#     accuracy = trainer.train_model()
    
#     # Show registration instructions
#     if accuracy > 0.7:  # Only register if model is decent
#         register_model_dagshub()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
import sklearn

class DiabetesModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_score = 0
        
        # Configuration - Use relative paths to avoid Windows path issues
        self.MLFLOW_TRACKING_URI = "mlruns"  # Relative path
        self.MLFLOW_EXPERIMENT_NAME = "diabetes_prediction"
        self.MODEL_DIR = "models"
        self.BEST_MODEL_NAME = "best_diabetes_model.pkl"
        self.DATA_PATH = "data/diabetes.csv"  # Relative path
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        
        # Create directories if they don't exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.MLFLOW_TRACKING_URI, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess the diabetes dataset"""
        print(f"Loading data from: {self.DATA_PATH}")
        df = pd.read_csv(self.DATA_PATH)
        
        # Handle missing values (zeros that should be NaN)
        columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_to_clean:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        return X, y
    
    def train_model(self):
        """Train the model with MLflow tracking"""
        # Set up MLflow
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.MLFLOW_EXPERIMENT_NAME)
        
        # Load and prepare data
        df = self.load_data()
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        with mlflow.start_run():
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.RANDOM_STATE
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            
            # Log parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("test_size", self.TEST_SIZE)
            mlflow.log_param("random_state", self.RANDOM_STATE)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", cr['1']['precision'])
            mlflow.log_metric("recall", cr['1']['recall'])
            mlflow.log_metric("f1_score", cr['1']['f1-score'])
            
            # Log model
            mlflow.sklearn.log_model(self.model, "random_forest_model")
            
            # Log artifacts
            mlflow.log_artifact(self.DATA_PATH)
            
            print(f"Model trained with accuracy: {accuracy:.4f}")
            print(f"Precision: {cr['1']['precision']:.4f}")
            print(f"Recall: {cr['1']['recall']:.4f}")
            print(f"F1-Score: {cr['1']['f1-score']:.4f}")
            
            # Save best model locally
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.save_model()
                
            return accuracy
    
    def save_model(self):
        """Save the best model locally"""
        model_path = f"{self.MODEL_DIR}/{self.BEST_MODEL_NAME}"
        
        # Create a more robust model dictionary
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'metadata': {
                'accuracy': self.best_score,
                'model_type': 'RandomForest',
                'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                'sklearn_version': sklearn.__version__,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            }
        }
        
        joblib.dump(model_dict, model_path)
        print(f"✅ Best model saved to {model_path}")

def register_model_dagshub():
    """Register the best model for DAGsHub (manual integration)"""
    mlflow.set_tracking_uri("mlruns")
    experiment_name = "diabetes_prediction"
    
    try:
        # Search for the best run
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print("No experiment found")
            return
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            print("No runs found in experiment")
            return
            
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        
        print(f"Best run ID: {best_run.run_id}")
        print(f"Best accuracy: {best_run['metrics.accuracy']:.4f}")
        
        # Model URI for registration
        model_uri = f"runs:/{best_run.run_id}/random_forest_model"
        
        print("=" * 50)
        print("DAGsHub Registration Instructions:")
        print("1. Set up your DAGsHub repository")
        print("2. Configure MLflow tracking URI:")
        print("   mlflow.set_tracking_uri('https://dagshub.com/<username>/<repository>.mlflow')")
        print("3. Use this model URI to register:")
        print(f"   Model URI: {model_uri}")
        print("4. Run: mlflow.register_model(model_uri, 'diabetes-prediction-model')")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error in model registration: {e}")

if __name__ == "__main__":
    # Train the model
    trainer = DiabetesModelTrainer()
    accuracy = trainer.train_model()
    
    # Show registration instructions
    if accuracy > 0.7:  # Only register if model is decent
        register_model_dagshub()