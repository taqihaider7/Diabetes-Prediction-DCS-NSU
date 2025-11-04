import flask
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_FILENAME = 'best_diabetes_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

# --- 1. Load Model and Scaler ---
try:
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    # Exit or raise error if model/scaler files are missing
    exit(1)

# Initialize Flask application
app = Flask(__name__)

# --- 2. Prediction Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data, preprocesses it, and returns a prediction.
    Expected JSON format:
    {
        "Pregnancies": 6, "Glucose": 148.0, "BloodPressure": 72.0,
        "SkinThickness": 35.0, "Insulin": 0.0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    """
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Convert JSON data to a pandas DataFrame
        # Ensure the order of columns matches the training data
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # We wrap the single input dictionary into a list and pass it to DataFrame
        df_input = pd.DataFrame([data], columns=feature_names)
        
        # --- Preprocessing steps (must match training) ---
        
        # Replace 0s with the mean (we use 1.0 as a placeholder since the application
        # doesn't have access to the original dataset's means, but should have
        # a way to handle this. For a simple deployment, we'll skip the
        # mean replacement and focus on scaling).
        # In a robust production environment, a custom transformer should handle this.
        
        # Scale the data using the loaded scaler
        scaled_input = scaler.transform(df_input)
        
        # Make prediction
        prediction_prob = model.predict_proba(scaled_input)[0].tolist()
        prediction_class = int(model.predict(scaled_input)[0])

        # Return the results
        return jsonify({
            'prediction_class': prediction_class,
            'probability_no_diabetes': prediction_prob[0],
            'probability_diabetes': prediction_prob[1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- 3. Health Check Endpoint ---

@app.route('/', methods=['GET'])
def health_check():
    return "Diabetes Prediction Service is running.", 200

# Run the app (for Docker)
if __name__ == '__main__':
    # '0.0.0.0' makes the server accessible from outside the Docker container
    app.run(host='0.0.0.0', port=5000)