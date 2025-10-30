"""
Streamlit Application for Diabetes Prediction using XGBoost Model
This application provides an interactive UI for diabetes prediction using the trained XGBoost model.

Features:
- Single patient prediction
- Batch predictions from CSV
- Model information and performance metrics
- Data visualization and statistics
- Confidence indicators and explanations
- Feature importance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple, List

# Configure page
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Feature Configuration ====================
FEATURE_COLUMNS = [
    'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N9', 'N10', 'N11', 'N15',
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'N0', 'N8', 'N13', 'N12', 'N14'
]

# Feature descriptions for user guidance
FEATURE_DESCRIPTIONS = {
    'Pregnancies': 'Number of times pregnant',
    'Glucose': 'Plasma glucose concentration (mg/dL)',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skin fold thickness (mm)',
    'Insulin': ' 2-Hour serum insulin (mu U/ml)',
    'BMI': 'Body mass index (kg/m²)',
    'DiabetesPedigreeFunction': 'Diabetes pedigree function score',
    'Age': 'Age in years',
    'N0': 'Engineered feature N0',
    'N1': 'Engineered feature N1',
    'N2': 'Engineered feature N2',
    'N3': 'Engineered feature N3',
    'N4': 'Engineered feature N4',
    'N5': 'Engineered feature N5',
    'N6': 'Engineered feature N6',
    'N7': 'Engineered feature N7',
    'N8': 'Engineered feature N8',
    'N9': 'Engineered feature N9',
    'N10': 'Engineered feature N10',
    'N11': 'Engineered feature N11',
    'N12': 'Engineered feature N12',
    'N13': 'Engineered feature N13',
    'N14': 'Engineered feature N14',
    'N15': 'Engineered feature N15'
}

# ==================== Cache and State Management ====================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                logger.info(f"Loading model from: {model_path}")
                model = joblib.load(model_path)
                logger.info(f"✓ Model loaded successfully")
                return model, model_path
            else:
                logger.error("No .joblib model files found")
                return None, None
        else:
            logger.error(f"Models directory not found: {models_dir}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_resource
def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """Load model metadata"""
    try:
        if not model_path:
            return {}
        
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"✓ Metadata loaded successfully")
                return metadata
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}


@st.cache_data
def load_processed_data_sample():
    """Load sample of processed data for reference"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"✓ Processed data loaded: {len(df)} rows")
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        return None


# ==================== Prediction Functions ====================
def make_prediction(features_dict: Dict[str, float]) -> Tuple[int, float, float]:
    """
    Make a single prediction
    Returns: (prediction, probability, confidence)
    """
    try:
        model, _ = load_model()
        
        if model is None:
            st.error("❌ Model not loaded. Please check the models directory.")
            return None, None, None
        
        # Prepare feature array in correct order
        feature_array = np.array([features_dict[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0][1]
        confidence = max(model.predict_proba(feature_array)[0])
        
        return int(prediction), float(probability), float(confidence)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None


def validate_features(features_dict: Dict[str, float]) -> Tuple[bool, str]:
    """Validate feature values"""
    try:
        # Age validation
        if features_dict['Age'] <= 0 or features_dict['Age'] > 150:
            return False, "❌ Age must be between 1 and 150 years"
        
        # BMI validation
        if features_dict['BMI'] <= 0:
            return False, "❌ BMI must be positive"
        
        # Medical value validations
        medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']
        for feature in medical_features:
            if features_dict[feature] < 0:
                return False, f"❌ {feature} cannot be negative"
        
        return True, "✓ Features validated successfully"
    
    except Exception as e:
        return False, f"❌ Validation error: {str(e)}"


# ==================== Streamlit UI ====================

# Custom CSS with Theme Support
st.markdown("""
    <style>
    /* Performance Metrics */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    .stMetric > div:nth-child(2) {
        color: #ffffff !important;
        font-size: 1.5em !important;
        font-weight: bold !important;
    }
    
    /* Prediction Result Boxes */
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-box h3 {
        margin-top: 0;
        font-size: 1.3em;
        margin-bottom: 10px;
    }
    
    .prediction-box p {
        margin: 5px 0;
        font-size: 0.95em;
    }
    
    /* High Risk Styling */
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-left: 6px solid #c92a2a;
        color: #ffffff;
    }
    
    .high-risk h3 {
        color: #ffffff;
    }
    
    .high-risk p {
        color: #f5f5f5;
    }
    
    /* Low Risk Styling */
    .low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        border-left: 6px solid #2f9e44;
        color: #ffffff;
    }
    
    .low-risk h3 {
        color: #ffffff;
    }
    
    .low-risk p {
        color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("# 🏥 Diabetes Prediction System")
st.markdown("**AI-powered diabetes risk prediction using Machine Learning**")
st.divider()

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Load model
model, model_path = load_model()
model_metadata = load_model_metadata(model_path) if model_path else {}
sample_data = load_processed_data_sample()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Application Settings")
    
    page = st.radio(
        "Select Page:",
        ["🔮 Single Prediction", "📊 Batch Prediction", "ℹ️ Model Info", "📈 Data Analysis"]
    )
    
    st.divider()
    
    # Model status
    st.markdown("### 📦 Model Status")
    if model is not None:
        st.success("✓ Model Loaded")
        st.caption(f"Model file: {os.path.basename(model_path) if model_path else 'N/A'}")
        if model_metadata:
            st.caption(f"Training date: {model_metadata.get('training_date', 'N/A')}")
            st.caption(f"Accuracy: {model_metadata.get('full_data_metrics', {}).get('accuracy', 'N/A')}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.divider()
    
    st.markdown("### 📋 Quick Guide")
    st.caption("""
    1. **Single Prediction**: Predict for one patient
    2. **Batch Prediction**: Upload CSV file with multiple patients
    3. **Model Info**: View model details and metrics
    4. **Data Analysis**: Explore data statistics
    """)


# ==================== PAGE 1: Single Prediction ====================
if page == "🔮 Single Prediction":
    st.markdown("## 🔮 Single Patient Prediction")
    st.markdown("Enter patient details below to get a diabetes risk prediction")
    st.divider()
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Medical History", "Measurements", "Engineered Features"])
    
    features_dict = {}
    
    # Tab 1: Medical History
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            features_dict['Pregnancies'] = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                help=FEATURE_DESCRIPTIONS['Pregnancies']
            )
            features_dict['Age'] = st.slider(
                "Age (years)",
                min_value=1,
                max_value=150,
                value=30,
                help=FEATURE_DESCRIPTIONS['Age']
            )
        
        with col2:
            features_dict['DiabetesPedigreeFunction'] = st.slider(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=2.5,
                step=0.01,
                value=0.5,
                help=FEATURE_DESCRIPTIONS['DiabetesPedigreeFunction']
            )
    
    # Tab 2: Physical Measurements
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            features_dict['BMI'] = st.slider(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=60.0,
                step=0.1,
                value=25.0,
                help=FEATURE_DESCRIPTIONS['BMI']
            )
            features_dict['BloodPressure'] = st.slider(
                "Blood Pressure (mm Hg)",
                min_value=0,
                max_value=200,
                value=70,
                help=FEATURE_DESCRIPTIONS['BloodPressure']
            )
            features_dict['SkinThickness'] = st.slider(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=20,
                help=FEATURE_DESCRIPTIONS['SkinThickness']
            )
        
        with col2:
            features_dict['Glucose'] = st.slider(
                "Glucose (mg/dL)",
                min_value=0,
                max_value=500,
                value=120,
                help=FEATURE_DESCRIPTIONS['Glucose']
            )
            features_dict['Insulin'] = st.slider(
                "Insulin (mu U/ml)",
                min_value=0,
                max_value=900,
                value=100,
                help=FEATURE_DESCRIPTIONS['Insulin']
            )
    
    # Tab 3: Engineered Features
    with tab3:
        st.info("💡 These are automatically engineered features derived from medical measurements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            features_dict['N0'] = st.number_input("Feature N0", value=0.0, step=0.01)
            features_dict['N1'] = st.number_input("Feature N1", value=0.0, step=0.01)
            features_dict['N2'] = st.number_input("Feature N2", value=0.0, step=0.01)
            features_dict['N3'] = st.number_input("Feature N3", value=0.0, step=0.01)
            features_dict['N4'] = st.number_input("Feature N4", value=0.0, step=0.01)
            features_dict['N5'] = st.number_input("Feature N5", value=0.0, step=0.01)
            features_dict['N6'] = st.number_input("Feature N6", value=0.0, step=0.01)
            features_dict['N7'] = st.number_input("Feature N7", value=0.0, step=0.01)
        
        with col2:
            features_dict['N8'] = st.number_input("Feature N8", value=0.0, step=0.01)
            features_dict['N9'] = st.number_input("Feature N9", value=0.0, step=0.01)
            features_dict['N10'] = st.number_input("Feature N10", value=0.0, step=0.01)
            features_dict['N11'] = st.number_input("Feature N11", value=0.0, step=0.01)
            features_dict['N12'] = st.number_input("Feature N12", value=0.0, step=0.01)
            features_dict['N13'] = st.number_input("Feature N13", value=0.0, step=0.01)
            features_dict['N14'] = st.number_input("Feature N14", value=0.0, step=0.01)
        
        with col3:
            features_dict['N15'] = st.number_input("Feature N15", value=0.0, step=0.01)
            
            st.markdown("**Load from Sample Data:**")
            if sample_data is not None and st.button("📥 Load Random Sample",
             width='stretch'):
                sample = sample_data.iloc[np.random.randint(0, len(sample_data))]
                for col in FEATURE_COLUMNS:
                    if col in sample.index:
                        st.session_state[f'input_{col}'] = float(sample[col])
                st.rerun()
    
    # Prediction section
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        if st.button("🔮 Make Prediction",
         width='stretch', type="primary"):
            # Validate features
            is_valid, validation_msg = validate_features(features_dict)
            
            if not is_valid:
                st.error(validation_msg)
            else:
                # Make prediction
                prediction, probability, confidence = make_prediction(features_dict)
                
                if prediction is not None:
                    st.session_state.prediction_made = True
                    st.session_state.last_prediction = {
                        'prediction': prediction,
                        'probability': probability,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
    
    with col3:
        if st.button("🔄 Reset Form",
         width='stretch'):
            st.rerun()
    
    # Display prediction results
    if st.session_state.prediction_made and st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        prediction = pred['prediction']
        probability = pred['probability']
        confidence = pred['confidence']
        
        st.divider()
        st.markdown("## 📊 Prediction Results")
        
        # Main result
        if prediction == 1:
            st.markdown("""
                <div class="prediction-box high-risk">
                <h3>⚠️ High Risk of Diabetes</h3>
                <p>The model predicts this patient has a high risk of diabetes.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-box low-risk">
                <h3>✅ Low Risk of Diabetes</h3>
                <p>The model predicts this patient has a low risk of diabetes.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
        
        with col2:
            st.metric("Probability", f"{probability:.2%}")
        
        with col3:
            st.metric("Confidence", f"{confidence:.2%}")
        
        with col4:
            st.metric("Risk Level", "High" if probability > 0.5 else "Low")
        
        # Probability gauge
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                title="Diabetes Risk (%)",
                delta={'reference': 50, 'relative': False},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig,
             width='stretch')
        
        with col2:
            # Class probability bar chart
            classes = ['No Diabetes', 'Diabetes']
            probs = [1 - probability, probability]
            colors = ['#00cc00', '#ff0000']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=probs,
                    y=classes,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f'{p:.2%}' for p in probs],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability",
                yaxis_title="Class",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig,
             width='stretch')
        
        # Timestamp
        st.caption(f"Prediction made at: {pred['timestamp']}")


# ==================== PAGE 2: Batch Prediction ====================
elif page == "📊 Batch Prediction":
    st.markdown("## 📊 Batch Prediction")
    st.markdown("Upload a CSV file with multiple patients for batch predictions")
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV file should contain columns matching the feature names"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File uploaded: {len(df)} rows")
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10),
             width='stretch')
            
            # Check for required columns
            missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.success("✓ All required columns present")
                
                # Make batch predictions
                if st.button("🔮 Make Batch Predictions", type="primary",
                 width='stretch'):
                    progress_bar = st.progress(0)
                    predictions_list = []
                    
                    for idx, row in df.iterrows():
                        try:
                            features = {col: row[col] for col in FEATURE_COLUMNS}
                            pred, prob, conf = make_prediction(features)
                            
                            predictions_list.append({
                                'Index': idx + 1,
                                'Prediction': 'Diabetes' if pred == 1 else 'No Diabetes',
                                'Probability': f'{prob:.4f}',
                                'Confidence': f'{conf:.4f}',
                                'Risk_Level': 'High' if prob > 0.5 else 'Low'
                            })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        except Exception as e:
                            logger.error(f"Error predicting row {idx}: {str(e)}")
                            predictions_list.append({
                                'Index': idx + 1,
                                'Prediction': 'Error',
                                'Probability': 'N/A',
                                'Confidence': 'N/A',
                                'Risk_Level': 'Error'
                            })
                    
                    # Display results
                    st.divider()
                    st.markdown("### 📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame(predictions_list)
                    st.dataframe(results_df,
                     width='stretch')
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    diabetes_count = sum(1 for p in predictions_list if p['Prediction'] == 'Diabetes')
                    high_risk_count = sum(1 for p in predictions_list if p['Risk_Level'] == 'High')
                    
                    with col1:
                        st.metric("Total Predictions", len(results_df))
                    with col2:
                        st.metric("Diabetes Cases", diabetes_count)
                    with col3:
                        st.metric("High Risk", high_risk_count)
                    with col4:
                        st.metric("Success Rate", f"{((len(results_df) - sum(1 for p in predictions_list if p['Prediction'] == 'Error')) / len(results_df) * 100):.1f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ==================== PAGE 3: Model Information ====================
elif page == "ℹ️ Model Info":
    st.markdown("## ℹ️ Model Information")
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Model Details")
        
        if model_metadata:
            st.markdown(f"""
            **Model Type:** {model_metadata.get('model_type', 'XGBoost')}
            
            **Training Date:** {model_metadata.get('training_date', 'N/A')}
            
            **Model Framework:** XGBoost
            
            **Feature Count:** {len(FEATURE_COLUMNS)}
            """)
        else:
            st.info("ℹ️ Metadata not available")
        
        st.markdown("### 📊 Performance Metrics")
        
        if model_metadata and 'full_data_metrics' in model_metadata:
            metrics = model_metadata['full_data_metrics']
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}")
                st.metric("Precision", f"{metrics.get('precision', 'N/A')}")
                st.metric("Recall", f"{metrics.get('recall', 'N/A')}")
            
            with col_m2:
                st.metric("F1 Score", f"{metrics.get('f1_score', 'N/A')}")
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
        else:
            st.info("ℹ️ Performance metrics not available")
    
    with col2:
        st.markdown("### 📝 Model Features")
        
        st.markdown("**Medical Features:**")
        medical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        for feat in medical_features:
            st.caption(f"• {feat}: {FEATURE_DESCRIPTIONS[feat]}")
        
        st.markdown("**Engineered Features (N0-N15):**")
        engineered_features = [f for f in FEATURE_COLUMNS if f.startswith('N')]
        for feat in engineered_features:
            st.caption(f"• {feat}: Automatically engineered feature")


# ==================== PAGE 4: Data Analysis ====================
elif page == "📈 Data Analysis":
    st.markdown("## 📈 Data Analysis")
    st.divider()
    
    if sample_data is not None:
        st.markdown(f"### Dataset Overview: {len(sample_data)} samples")
        
        # Basic statistics
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(sample_data[FEATURE_COLUMNS].describe(),
        
        
        # Visualizations
        st.markdown("### 📈 Feature Distributions")
        
        # Select features to visualize
        selected_features = st.multiselect(
            "Select features to visualize:",
            FEATURE_COLUMNS,
            default=['Glucose', 'BMI', 'Age', 'BloodPressure']
        )
        
        if selected_features:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograms
                for feat in selected_features[:2]:
                    fig = px.histogram(
                        sample_data,
                        x=feat,
                        nbins=30,
                        title=f"Distribution of {feat}",
                        labels={feat: feat}
                    )
                    st.plotly_chart(fig,
                     width='stretch')
            
            with col2:
                # Box plots
                for feat in selected_features[2:]:
                    fig = px.box(
                        sample_data,
                        y=feat,
                        title=f"Box Plot of {feat}"
                    )
                    st.plotly_chart(fig,
                     width='stretch')
        
        # Correlation analysis
        st.markdown("### 🔗 Feature Correlations")
        
        if st.checkbox("Show correlation heatmap"):
            fig = px.imshow(
                sample_data[FEATURE_COLUMNS].corr(),
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto",
                height=800
            )
            st.plotly_chart(fig,
             width='stretch')
    else:
        st.warning("⚠️ Processed data not available for analysis")


# Footer
st.divider()
st.markdown("""
    ---
    **Diabetes Prediction System v1.0** | Built with Streamlit & XGBoost
    
    *Disclaimer: This application is for educational purposes. Always consult with healthcare professionals for medical decisions.*
""")
