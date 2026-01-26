import pandas as pd
import pickle
import os
import streamlit as st

# Constants
DATA_PATH = 'data/procressed/elliptic_bitcoin_dataset_v1_graph.csv'
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
