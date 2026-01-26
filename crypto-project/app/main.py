import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, load_models, get_feature_importance

# Page Config
st.set_page_config(
    page_title="Crypto Fraud Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Style
st.title("🕵️‍♂️ Bitcoin Transaction Fraud Detection")
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# sidebar
st.sidebar.header("⚙️ Configuration")

# Load Data
with st.spinner('Loading Dataset...'):
    df = load_data()
    models = load_models()

if df is not None and models:
    # 1. Dashboard Overview
    st.header("1. 📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    total_tx = len(df)
    illicit_tx = len(df[df['label'] == 1])
    licit_tx = len(df[df['label'] == 0])
    
    with col1:
        st.metric("Total Transactions", f"{total_tx:,}")
    with col2:
        st.metric("Licit Transactions (Safe)", f"{licit_tx:,}", delta_color="normal")
    with col3:
        st.metric("Illicit Transactions (Fraud)", f"{illicit_tx:,}", delta_color="inverse")

    # Transaction Distribution Plot
    fig_dist = px.pie(
        names=['Licit', 'Illicit'],
        values=[licit_tx, illicit_tx],
        color=['Licit', 'Illicit'],
        color_discrete_map={'Licit':'#00CC96', 'Illicit':'#EF553B'},
        title="Transaction Class Distribution"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # 2. Prediction Interface
    st.header("2. 🔍 Fraud Prediction")
    
    st.sidebar.subheader("Model Selection")
    selected_model_name = st.sidebar.selectbox("Choose Classifier", list(models.keys()))
    model = models[selected_model_name]

    # Test Sample Selection
    st.subheader("Test a Transaction")
    
    # Filter only test set (time_step > 34) for realistic simulation
    test_ios = df[df['time_step'] > 34]
    
    # Let user pick a random sample or specific ID
    sample_method = st.radio("Select Input Method", ["Random Sample", "Select by ID"])
    
    selected_tx = None
    
    if sample_method == "Random Sample":
        if st.button("🎲 Get Random Transaction"):
            selected_tx = test_ios.sample(1)
            st.session_state['selected_tx'] = selected_tx
        elif 'selected_tx' in st.session_state:
            selected_tx = st.session_state['selected_tx']
    else:
        tx_id_input = st.number_input("Enter Transaction ID (txId)", min_value=0)
        if tx_id_input in test_ios['txId'].values:
            selected_tx = test_ios[test_ios['txId'] == tx_id_input]
        elif tx_id_input != 0:
            st.error("Transaction ID not found in Test Set.")

    if selected_tx is not None:
        st.markdown("### Transaction Details")
        st.dataframe(selected_tx.head())
        
        # Prepare features for prediction (drop metadata)
        # Assuming columns to drop are label, txId, and potentially time_step if model trained without it
        # Based on evaluation notebook: drop_cols = ['label', 'txId', 'time_step']
        # But we need to check if 'class' exists (original dataset)
        
        features = selected_tx.drop(columns=['label', 'txId', 'time_step'], errors='ignore')
        if 'class' in features.columns:
            features = features.drop(columns=['class'])
            
        if st.button("🚀 Predict"):
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error("### 🚨 FRAUD DETECTED")
                else:
                    st.success("### ✅ SAFE TRANSACTION")
            
            with col_res2:
                st.metric("Fraud Probability", f"{probability:.2%}")
                st.progress(float(probability))
            
            # Show Ground Truth Comparison
            actual_label = selected_tx['label'].values[0]
            st.markdown("---")
            st.subheader("🎯 Prediction Accuracy")
            
            col_truth1, col_truth2 = st.columns(2)
            
            with col_truth1:
                if actual_label == 1:
                    st.warning("⚠️ Actual Status: **FRAUD**")
                else:
                    st.info("ℹ️ Actual Status: **SAFE**")
            
            with col_truth2:
                if prediction == actual_label:
                    st.success("### ✅ Prediction is CORRECT")
                else:
                    st.error("### ❌ Prediction is WRONG")

            # 3. Explainability (Feature Importance for this decision)
            # Simple global feature importance for the selected model
            st.subheader("3. 🧠 Model Explanation")
            
            feature_imp = get_feature_importance(model, features.columns)
            
            if feature_imp is not None:
                top_n = 10
                fig_imp = px.bar(
                    feature_imp.head(top_n),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top {top_n} Factors for {selected_model_name}",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
                
                
                # Highlight graph features
                graph_feats = [f for f in feature_imp['Feature'].head(20).values if 'degree' in f]
                if graph_feats:
                    st.info(f"💡 Key Graph Features used: {', '.join(graph_feats)}")
                
                # --- NEW GRAPH 1: Radar Chart (Comparison with Average Fraud) ---
                st.subheader("4. 🕸️ Fraud Profile Comparison (Radar Chart)")
                st.markdown("Comparing this transaction's features with the **Average Illicit Transaction** profile.")
                
                # Select top 5 important features to plot
                top_features = feature_imp['Feature'].head(5).tolist()
                
                # Normalize data for Radar Chart using Percentile Rank
                # This ensures the chart is readable even with outliers
                subset_df = df[top_features]
                
                # Calculate percentile rank for the whole dataset
                normalized_df = subset_df.rank(pct=True)
                
                # Get Avg Fraud Profile (Normalized)
                fraud_indices = df[df['label'] == 1].index
                avg_fraud = normalized_df.loc[fraud_indices].mean()
                
                # Get Selected Transaction (Normalized)
                # Ensure we handle the single row selection correctly
                tx_index = selected_tx.index[0]
                current_tx_norm = normalized_df.loc[tx_index]
                
                categories = top_features
                
                
                fig_radar = go.Figure()
                
                # Neon Red for Fraud Profile
                fig_radar.add_trace(go.Scatterpolar(
                    r=avg_fraud.values,
                    theta=categories,
                    fill='toself',
                    name='Average Fraud Profile',
                    line=dict(color='#FF4B4B', width=3),
                    fillcolor='rgba(255, 75, 75, 0.3)',
                    marker=dict(size=8)
                ))
                
                # Neon Blue/Cyan for Current Transaction
                fig_radar.add_trace(go.Scatterpolar(
                    r=current_tx_norm.values,
                    theta=categories,
                    fill='toself',
                    name='This Transaction',
                    line=dict(color='#00CC96', width=3),
                    fillcolor='rgba(0, 204, 150, 0.3)',
                    marker=dict(size=10, symbol='diamond')
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='gray'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=True,
                    title="Feature Values (Percentile Rank)",
                    template='plotly_dark', # Force dark theme compatible colors
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # --- NEW GRAPH 2: Feature Distribution (Box Plot) ---
                st.subheader(f"5. 📉 Feature Position: {top_features[0]}")
                st.markdown(f"Where does this transaction fall for the most important feature (**{top_features[0]}**) compared to others?")
                
                # Prepare data for plotting
                plot_data = df[['label', top_features[0]]].copy()
                plot_data['Type'] = plot_data['label'].map({0: 'Licit', 1: 'Illicit'})
                
                fig_box = px.box(plot_data, x='Type', y=top_features[0], color='Type', 
                                color_discrete_map={'Licit':'#00CC96', 'Illicit':'#FF4B4B'},
                                points=False) 
                
                # Add point for current transaction
                current_val = selected_tx[top_features[0]].values[0]
                
                fig_box.add_trace(go.Scatter(
                    x=['Illicit' if prediction == 1 else 'Licit'], 
                    y=[current_val],
                    mode='markers+text',
                    name='This Transaction',
                    text=['⬅ THIS TX'],
                    textposition="middle right",
                    marker=dict(color='#FFFFFF', size=15, symbol='star', line=dict(color='#FFFF00', width=2))
                ))
                
                fig_box.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)

            else:
                st.info("Feature importance not available for this model type.")
                
else:
    st.info("Please generate the dataset and models first using the notebooks.")
