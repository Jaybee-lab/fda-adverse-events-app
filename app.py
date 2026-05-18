import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.title("FDA Adverse Event Risk Predictor")
st.write("This tool uses machine learning to predict if a drug side effect report is likely to be classified as 'Serious'.")

# 1. User Input Section
st.header("Patient & Clinical Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age", value=55.9) # Median from source [cite: 317]
    sex = st.selectbox("Patient Sex", ["Female", "Male", "Unknown"])
    weight = st.number_input("Patient Weight (kg)", value=74.3) # Median from source [cite: 317]

with col2:
    num_drugs = st.number_input("Number of Drugs Taken", min_value=1, value=8) # Mean from source [cite: 317]
    num_reactions = st.number_input("Number of Reactions Reported", min_value=1, value=6) # Mean from source [cite: 317]

brand_name = st.text_input("Drug Brand Name", "Unknown")
pharm_class = st.text_input("Pharmacological Class", "Unknown")

# 2. Prediction Logic
if st.button("Analyze Report"):
    # Prepare data for the model
    input_df = pd.DataFrame([{
        'year': 2026, 'month': 5, 
        'patient_age_years': age, 'patient_weight_kg': weight,
        'patient_sex': sex, 'brand_name': brand_name,
        'pharm_class': pharm_class, 'num_drugs': num_drugs,
        'num_reactions': num_reactions
    }])

    # Apply Encoding and Scaling as done in Phase 3 [cite: 691, 704]
    input_df[['patient_sex', 'brand_name', 'pharm_class']] = encoder.transform(
        input_df[['patient_sex', 'brand_name', 'pharm_class']].astype(str)
    )
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    # 3. Display Results
    if prediction[0] == 1:
        st.error(f"HIGH RISK: This event is predicted to be SERIOUS (Confidence: {probability:.2%})")
    else:
        st.success(f"LOW RISK: This event is predicted to be NON-SERIOUS (Confidence: {1-probability:.2%})")