import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('fda_random_forest_model.pkl')
scaler = joblib.load('fda_scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.set_page_config(page_title="FDA Adverse Event Predictor", layout="centered")

st.title("🏥 FDA Adverse Event Seriousness Predictor")
st.markdown("This tool predicts whether a reported drug side effect is likely to be **Serious** based on patient and drug data.")

with st.form("prediction_form"):
    st.subheader("Patient & Event Information")
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Reporting Year", min_value=2015, max_value=2026, value=2024)
        month = st.slider("Month", 1, 12, 1)
        age = st.number_input("Patient Age (Years)", 0, 110, 45)
        weight = st.number_input("Patient Weight (kg)", 1.0, 400.0, 70.0)
        sex = st.selectbox("Patient Sex", ["Male", "Female", "Unknown"])

    with col2:
        brand_name = st.text_input("Brand Name", "Generic")
        pharm_class = st.text_input("Pharmacological Class", "Unknown")
        num_drugs = st.number_input("Number of Concurrent Drugs", 1, 50, 1)
        num_reactions = st.number_input("Number of Reactions Reported", 1, 20, 1)
    
    submit = st.form_submit_button("Predict Seriousness")

if submit:
    cat_data = pd.DataFrame([[sex, brand_name, pharm_class, "Unknown"]], 
                            columns=['patient_sex', 'brand_name', 'pharm_class', 'age_group'])
    
    encoded_cats = encoder.transform(cat_data.astype(str))
    

    features = np.array([[
        year, month, age, weight, 
        encoded_cats[0][0], 
        encoded_cats[0][1], 
        encoded_cats[0][2], 
        num_drugs, 
        num_reactions
    ]])
    
  
    scaled_features = scaler.transform(features)
    
  
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"### Prediction: SERIOUS EVENT")
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.success(f"### Prediction: NON-SERIOUS EVENT")
        st.write(f"Confidence Score: {(1-probability):.2%}")