import pandas as pd
import pickle
import os
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Constants
DATA_PATH = 'data/processed/elliptic_bitcoin_dataset_v1_graph.csv'
MODELS_DIR = 'models'

@st.cache_data
def load_data():
    """Loads the V1 graph dataset efficiently."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # Rename for clarity if needed
            if 'feat_0' in df.columns:
                df = df.rename(columns={'feat_0': 'time_step'})
            return df
        else:
            st.error(f"Dataset not found at {DATA_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Loads all available V1 models from the models directory."""
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression_smote_v1_graph.pkl",
        "Random Forest": "random_forest_smote_v1_graph.pkl",
        "XGBoost": "xgboost_smote_v1_graph.pkl"
    }

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                models[name] = pickle.load(file)
        else:
            st.warning(f"Model file {filename} not found.")
    
    return models

def get_feature_importance(model, feature_names):
    """Extracts feature importance if the model supports it."""
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For Logistic Regression (using absolute coefficients)
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(model.coef_[0])
        }).sort_values(by='Importance', ascending=False)
    return None


def get_model_metrics(models, X_test, y_test):
    """
    Calculate performance metrics for all models.
    Returns a DataFrame with Accuracy, Precision, Recall, F1 for each model.
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        })
    
    return pd.DataFrame(results)


def get_confusion_matrix_data(model, X_test, y_test):
    """
    Generate confusion matrix data for a given model.
    Returns the confusion matrix as a numpy array.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm


def create_export_data(tx_data, prediction, probability, model_name):
    """
    Create a DataFrame for exporting prediction results.
    """
    export_df = tx_data.copy()
    export_df['Prediction'] = 'Fraud' if prediction == 1 else 'Safe'
    export_df['Fraud_Probability'] = probability
    export_df['Model_Used'] = model_name
    return export_df


# ============================================
# SHAP Analysis Functions
# ============================================
import shap

@st.cache_data
def get_shap_global_values(_model, X_sample, _model_name):
    """
    Calculate SHAP values for global feature importance.
    Uses a sample of 500 rows for speed.
    _model and _model_name are unhashable-prefixed to allow caching.
    """
    try:
        # Sample background data for speed
        if len(X_sample) > 500:
            X_bg = X_sample.sample(500, random_state=42)
        else:
            X_bg = X_sample
        
        # Choose explainer based on model type
        if hasattr(_model, 'feature_importances_'):
            # Tree-based models (Random Forest, XGBoost)
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(X_bg)
            # For binary classification, take class 1 (Fraud)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif hasattr(_model, 'coef_'):
            # Linear models (Logistic Regression)
            explainer = shap.LinearExplainer(_model, X_bg)
            shap_values = explainer.shap_values(X_bg)
        else:
            return None, None
        
        # Calculate mean absolute SHAP values per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X_bg.columns.tolist(),
            'SHAP_Importance': mean_shap
        }).sort_values('SHAP_Importance', ascending=False)
        
        return feature_importance, explainer
    except Exception as e:
        st.warning(f"SHAP calculation error: {e}")
        return None, None


def get_shap_single_values(model, features, X_background, model_name):
    """
    Calculate SHAP values for a single transaction prediction.
    Returns shap_values array and base_value for waterfall plot.
    """
    try:
        # Sample background data
        if len(X_background) > 100:
            X_bg = X_background.sample(100, random_state=42)
        else:
            X_bg = X_background
        
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_bg)
            shap_values = explainer.shap_values(features)
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
        else:
            return None, None
        
        return shap_values[0], base_value
    except Exception as e:
        st.warning(f"SHAP single prediction error: {e}")
        return None, None
