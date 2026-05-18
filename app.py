import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and scaler
@st.cache_resource
def load_assets():
    # Make sure these filenames match exactly what you downloaded from Colab
    model = joblib.load('fda_rf_model.pkl')
    scaler = joblib.load('fda_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("✅ Model Assets Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Make sure 'fda_rf_model.pkl' and 'fda_scaler.pkl' are in the same folder as app.py")

# 2. App Header
st.title("🏥 FDA Adverse Events Risk Predictor")
st.markdown("Predicts if a report will be **Serious** based on patient data.")

# 3. Sidebar Inputs
st.sidebar.header("📋 Input Patient Data")
year = st.sidebar.slider("Year", 2015, 2030, 2026)
month = st.sidebar.slider("Month", 1, 12, 1)
quarter = st.sidebar.slider("Quarter", 1, 4, 1)
age = st.sidebar.number_input("Age", 0, 110, 50)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "Generic")
p_class = st.sidebar.text_input("Pharm Class", "Unknown")
n_drugs = st.sidebar.number_input("Num Drugs", 1, 50, 2)
n_react = st.sidebar.number_input("Num Reactions", 1, 50, 1)

# 4. Prediction Logic
if st.button("🔮 Run Prediction"):
    # Encoding
    sex_enc = 0 if sex == "Female" else (1 if sex == "Male" else 2)
    brand_enc = abs(hash(brand)) % 1000
    class_enc = abs(hash(p_class)) % 1000
    
    # This part was likely causing your crash - keep it on one line or use proper brackets
    input_data = np.array([[year, month, quarter, age, weight, sex_enc, brand_enc, class_enc, n_drugs, n_react]])
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]
    
    # Display Results
    st.subheader("📊 Analysis Results")
    if prediction == 1:
        st.error(f"⚠️ **SERIOUS OUTCOME PREDICTED** (Probability: {proba:.2%})")
    else:
        st.success(f"✅ **NOT SERIOUS** (Probability of seriousness: {proba:.2%})")