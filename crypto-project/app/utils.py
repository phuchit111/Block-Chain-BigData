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
