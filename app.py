import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Assets
@st.cache_resource
def load_assets():
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
st.markdown("Use this tool to predict if a drug reaction is likely to be **Serious**.")

# 3. Sidebar Inputs
st.sidebar.header("Patient & Clinical Data")
year = st.sidebar.slider("Year", 2015, 2030, 2026)
month = st.sidebar.slider("Month", 1, 12, 1)
age = st.sidebar.number_input("Patient Age", 0, 110, 50)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "Generic")
p_class = st.sidebar.text_input("Pharm Class", "Unknown")
n_drugs = st.sidebar.number_input("Number of Drugs Taken", 1, 50, 2)
n_react = st.sidebar.number_input("Number of Reactions", 1, 50, 1)

# 4. Prediction Logic
if st.button("Run Risk Analysis"):
    if model and scaler:
        # Encoding (Must match your Colab training order)
        sex_enc = 0 if sex == "Female" else (1 if sex == "Male" else 2)
        brand_enc = abs(hash(brand)) % 1000
        class_enc = abs(hash(p_class)) % 1000
        
        # EXACT 9 FEATURES in CORRECT ORDER
        input_data = np.array([[year, month, age, weight, sex_enc, brand_enc, class_enc, n_drugs, n_react]])
        
        try:
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            proba = model.predict_proba(scaled_data)[0][1]
            
            st.divider()
            if prediction == 1:
                st.error(f"### ⚠️ HIGH RISK: SERIOUS OUTCOME PREDICTED")
                st.write(f"Confidence Level: **{proba:.2%}**")
            else:
                st.success(f"### ✅ LOW RISK: NOT SERIOUS")
                st.write(f"Probability of Seriousness: **{proba:.2%}**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# 5. Visualizations Section (The missing part)
st.divider()
st.header("📊 Model Insights")
tab1, tab2 = st.tabs(["Feature Importance", "Project Context"])

with tab1:
    st.subheader("Which factors drive the risk?")
    if model:
        importances = model.feature_importances_
        feature_names = ['Year', 'Month', 'Age', 'Weight', 'Sex', 'Brand', 'Class', 'Drugs', 'Reactions']
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, palette='magma', ax=ax)
        st.pyplot(fig)
        st.info("The chart shows that 'Number of Reactions' and 'Number of Drugs' are the strongest predictors of seriousness.")

with tab2:
    st.subheader("Project Summary")
    st.write("This model was trained on **528,000 FDA reports** using the CRISP-DM methodology.")
    st.write("- **Best Model:** Random Forest")
    st.write("- **Recall Score:** 0.7486 (Catches ~75% of serious cases)")