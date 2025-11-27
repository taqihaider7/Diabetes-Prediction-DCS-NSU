
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
import datetime

warnings.filterwarnings('ignore')

class DiabetesModelTrainer:
    def __init__(self, experiment_name="Diabetes-Experiments"):
        self.experiment_name = experiment_name
        self.models = {
            "random_forest": RandomForestClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42)
        }
        self.scaler = StandardScaler()
        
        # Setup MLflow tracking
        self.setup_mlflow_tracking()
    
    def setup_mlflow_tracking(self):
        """Setup MLflow tracking with DAGsHub"""
        # DAGsHub configuration
        DAGSHUB_USERNAME = "nawazishpatana"
        DAGSHUB_TOKEN = "a3e0c3abf610a86cea1f349c92b841da5816eac3"
        DAGSHUB_TRACKING_URI = "https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow"
        
        # Set environment variables for DAGsHub
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        
        # Set tracking URI
        mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
        
        print(f"🔧 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"🎯 Experiment: {self.experiment_name}")
    
    def load_data(self):
        """
        Load diabetes dataset from CSV file
        """
        try:
            df = pd.read_csv('data/diabetes.csv')
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            print(f"📊 Diabetes prevalence: {df['Outcome'].mean():.2%}")
            
            # Display basic info about the dataset
            print(f"📋 Features: {list(df.columns)}")
            print(f"🔍 First few rows:")
            print(df.head())
            
            return df
        except FileNotFoundError:
            print("❌ diabetes.csv not found. Creating synthetic data...")
            return self.create_synthetic_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """
        Create synthetic diabetes classification dataset as fallback
        """
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Pregnancies': np.random.poisson(2, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(80, 12, n_samples),
            'SkinThickness': np.random.normal(25, 8, n_samples),
            'Insulin': np.random.normal(100, 50, n_samples),
            'BMI': np.random.normal(28, 6, n_samples),
            'DiabetesPedigreeFunction': np.random.normal(0.5, 0.2, n_samples),
            'Age': np.random.normal(50, 15, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on features
        diabetes_risk = (
            0.1 * (df['Age'] - 50) / 15 +
            0.3 * (df['Glucose'] - 120) / 30 +
            0.2 * (df['BMI'] - 28) / 6 +
            0.1 * (df['BloodPressure'] - 80) / 12 +
            0.2 * (df['DiabetesPedigreeFunction'] - 0.5) / 0.2 +
            0.1 * np.random.normal(0, 1, n_samples)
        )
        
        df['Outcome'] = (diabetes_risk > 0.5).astype(int)
        
        print(f"📋 Synthetic dataset created: {df.shape[0]} samples")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data for training
        """
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def train_models(self):
        """
        Train multiple models and log to MLflow (DAGsHub compatible version)
        """
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Load and preprocess data
        print("📥 Loading and preprocessing data...")
        df = self.load_data()
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(df)
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        # Train each model and log to MLflow
        for model_name, model in self.models.items():
            print(f"\n🏋️ Training {model_name}...")
            
            # Create unique run name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_name}_{timestamp}"
            
            with mlflow.start_run(run_name=run_name):
                print(f"  🚀 Starting MLflow run: {run_name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Log parameters
                mlflow.log_params({
                    "model_type": model_name,
                    "random_state": 42,
                    "n_features": X_train.shape[1],
                    "n_samples": len(X_train),
                    "test_size": 0.2,
                    "features": ", ".join(feature_names)
                })
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                    print(f"  📈 {metric_name}: {metric_value:.4f}")
                
                # ✅ FIX: Use DAGsHub compatible model logging (without registered_model_name)
                try:
                    # Log model without registration (DAGsHub doesn't support model registry)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"model_{model_name}",
                        # Remove registered_model_name parameter for DAGsHub compatibility
                    )
                    print(f"  ✅ Model logged to MLflow")
                except Exception as e:
                    print(f"  ⚠️  Could not log model to MLflow: {e}")
                    print(f"  💾 Saving model locally only")
                
                # Log feature importance if available
                if hasattr(model, "feature_importances_"):
                    try:
                        importance_dict = dict(zip(feature_names, model.feature_importances_))
                        for feature, importance in importance_dict.items():
                            mlflow.log_metric(f"feature_importance_{feature}", importance)
                        print(f"  📊 Feature importance logged")
                    except Exception as e:
                        print(f"  ⚠️  Could not log feature importance: {e}")
                
                # Log dataset info
                try:
                    dataset_info = {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "positive_class_ratio": y_train.mean(),
                        "feature_names": list(feature_names)
                    }
                    mlflow.log_dict(dataset_info, "dataset_info.json")
                except Exception as e:
                    print(f"  ⚠️  Could not log dataset info: {e}")
                
                # Track best model
                current_score = metrics['f1_score']
                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_model_name = model_name
                    
                print(f"  ✅ {model_name} training completed")
        
        # Save best model locally
        if best_model is not None:
            self.save_best_model(best_model, best_model_name, best_score, self.scaler, feature_names)
        
        return best_model, best_model_name, best_score
    
    def save_best_model(self, model, model_name, score, scaler, feature_names):
        """
        Save the best model and scaler locally
        """
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save model and scaler together
        model_data = {
            'model': model,
            'scaler': scaler,
            'metadata': {
                'model_name': model_name,
                'f1_score': score,
                'feature_names': list(feature_names),
                'timestamp': datetime.datetime.now().isoformat(),
                'saved_locally': True
            }
        }
        
        model_path = "models/best_diabetes_model.pkl"
        joblib.dump(model_data, model_path)
        
        # Also save scaler separately for convenience
        scaler_path = "models/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"\n💾 Best model saved locally:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Type: {model_name}")
        print(f"   F1-Score: {score:.4f}")
        print(f"   Features: {', '.join(feature_names)}")
    
    def verify_mlflow_setup(self):
        """Verify MLflow is setup correctly"""
        print("\n🔍 Verifying MLflow Setup:")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   Experiment: {self.experiment_name}")
        
        try:
            # Try to get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                print(f"   ✅ Experiment found: {experiment.experiment_id}")
            else:
                print("   ℹ️  Experiment will be created on first run")
        except Exception as e:
            print(f"   ⚠️  Note: {e}")

def main():
    """
    Main function to run the training pipeline
    """
    # Initialize trainer
    trainer = DiabetesModelTrainer(experiment_name="Nawaz-Experiments")
    
    print("🚀 Starting Diabetes Prediction Model Training")
    print("=" * 50)
    
    # Verify MLflow setup first
    trainer.verify_mlflow_setup()
    
    try:
        # Train models and log to MLflow
        print("\n1. Training models with MLflow logging...")
        best_model, best_model_name, best_score = trainer.train_models()
        
        print(f"\n🎉 Training completed!")
        print(f"   Best model: {best_model_name}")
        print(f"   Best F1-Score: {best_score:.4f}")
        
        print(f"\n📊 View your experiments at:")
        print(f"   https://dagshub.com/taqihaider7/Diabetes-Prediction-DCS-NSU.mlflow")
        
        print(f"\n🔮 Next steps:")
        print(f"   1. Run: python src/app.py")
        print(f"   2. Visit: http://localhost:8000/docs")
        print(f"   3. Test your API endpoints!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"💡 Try running with local MLflow tracking:")
        print(f"   mlflow.set_tracking_uri('file:///./mlruns')")

if __name__ == "__main__":
    main()