import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go  # type: ignore[import]

# 1. Load Assets
@st.cache_resource
def load_assets():
    # Ensure these files are in the same folder as app.py
    model = joblib.load('fda_random_forest_model.pkl')
    scaler = joblib.load('fda_scaler.pkl')
    return model, scaler

model, scaler = None, None
try:
    model, scaler = load_assets()
    st.sidebar.success("Model Assets Loaded!")
except Exception as e:
    st.sidebar.error(f"Could not load model files: {e}")

# 2. UI Header
st.title("🛡️ FDA Adverse Event Risk Predictor")
st.markdown("Predicting safety signals from **2015-2025** data.")

# 3. Sidebar Inputs
st.sidebar.header("Patient & Clinical Data")
# Updated max_value to 2025 to match verifiable dataset range
year = st.sidebar.slider("Year", 2015, 2025, 2024) 
month = st.sidebar.slider("Month", 1, 12, 1)
age = st.sidebar.number_input("Patient Age", 0, 110, 50)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "HUMIRA")
p_class = st.sidebar.text_input("Pharm Class", "TNF Blocker")
n_drugs = st.sidebar.number_input("Number of Drugs Taken", 1, 50, 2)
n_react = st.sidebar.number_input("Number of Reactions", 1, 50, 1)

# 4. Prediction Logic & "Moving" Visuals
if st.button("Run Risk Analysis"):
    if model and scaler:
        sex_enc = 0 if sex == "Female" else (1 if sex == "Male" else 2)
        # Using a consistent hash for demonstration
        brand_enc = abs(hash(brand)) % 1000
        class_enc = abs(hash(p_class)) % 1000
        
        input_data = np.array([[year, month, age, weight, sex_enc, brand_enc, class_enc, n_drugs, n_react]])
        
        try:
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            proba_serious = model.predict_proba(scaled_data)[0][1]
            
            # --- MOVING VISUAL 1: Probability Gauge ---
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba_serious * 100,
                title = {'text': "Seriousness Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "red"}]}))
            st.plotly_chart(fig_gauge)

            if prediction == 1:
                st.error(f"### ⚠️ HIGH RISK PREDICTED")
            else:
                st.success(f"### ✅ LOW RISK PREDICTED")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# 5. Dynamic Insights Section
st.divider()
st.header("📊 Interactive Insights")
tab1, tab2 = st.tabs(["Feature Importance", "Project Context"])

with tab1:
    st.subheader("Which factors drive the risk?")
    if model:
        importances = model.feature_importances_
        feature_names = ['Year', 'Month', 'Age', 'Weight', 'Sex', 'Brand', 'Class', 'Drugs', 'Reactions']
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        # --- MOVING VISUAL 2: Interactive Bar Chart ---
        fig_importance = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Magma')
        st.plotly_chart(fig_importance)
        st.info("Hover over the bars to see exact importance values.")

with tab2:
    st.subheader("Project Summary")
    st.write("Trained on **528,000 FDA reports**.")
    st.write(f"- **Recall:** 0.7486")
    st.write(f"- **Accuracy:** 0.7182")