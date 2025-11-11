import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# --- Configuration ---
DATA_FILE = 'data/diabetes.csv'
TARGET_COLUMN = 'Outcome'
MODEL_FILENAME = 'Docker/best_diabetes_model.pkl'

# --- 1. Load and Prepare Data ---
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Columns where 0 is an indicator of missing data (not a true value)
cols_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with the mean of the column
print("Cleaning data (replacing 0s in key columns with the mean)...")
for col in cols_to_replace_zero:
    df[col] = df[col].replace(0, df[col].mean())

# Separate features (X) and target (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler object
with open('Docker/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Saved feature scaler to 'scaler.pkl'")

# --- 3. Model Experimentation and Selection ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine (SVM)': SVC(random_state=42, probability=True)
}

best_model = None
best_accuracy = 0.0
best_model_name = ""
results = {}

print("\n--- Starting Model Training and Evaluation ---")
for name, model in models.items():
    print(f"Training {name}...")
    
    # Use scaled data for LR and SVM, unscaled for RF
    X_train_data = X_train_scaled if name in ['Logistic Regression', 'Support Vector Machine (SVM)'] else X_train
    X_test_data = X_test_scaled if name in ['Logistic Regression', 'Support Vector Machine (SVM)'] else X_test

    model.fit(X_train_data, y_train)
    y_pred = model.predict(X_test_data)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"  {name} Accuracy: {accuracy:.4f}")
    
    # Check for the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\n--- Experiment Summary ---")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f}")

# --- 4. Model Serialization (Saving the Best Model) ---
print(f"\nBest Model Selected: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model using pickle
with open(MODEL_FILENAME, 'wb') as file:
    pickle.dump(best_model, file)
    
print(f"Successfully saved the best model to '{MODEL_FILENAME}'")
print(f"Saved the scaler to 'scaler.pkl'")