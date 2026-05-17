import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and scaler
@st.cache_resource
def load_assets():
    # Use the filenames you saved in Colab
    model = joblib.load('fda_random_forest_model.pkl')
    scaler = joblib.load('fda_scaler.pkl')
    return model, scaler

model, scaler = None, None

try:
    model, scaler = load_assets()
    st.success("Model Assets Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading assets: {e}")

st.title("FDA Adverse Events Risk Predictor")
st.markdown("Predicts if a report will be **Serious** based on patient data.")

# 2. Sidebar Inputs
st.sidebar.header("Input Patient Data")
year = st.sidebar.slider("Year", 2015, 2030, 2026)
month = st.sidebar.slider("Month", 1, 12, 1)
# Note: Quarter is kept for UI but NOT passed to the model
quarter = st.sidebar.slider("Quarter", 1, 4, 1) 
age = st.sidebar.number_input("Age", 0, 110, 50)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "Generic")
p_class = st.sidebar.text_input("Pharm Class", "Unknown")
n_drugs = st.sidebar.number_input("Num Drugs", 1, 50, 2)
n_react = st.sidebar.number_input("Num Reactions", 1, 50, 1)

# 3. Prediction Logic
if st.button("Run Prediction"):
    if model is None or scaler is None:
        st.error("Assets not loaded.")
    else:
        # Encoding logic (Must match your OrdinalEncoder training values)
        # 0: Female, 1: Male, 2: Unknown (based on your Colab setup)
        sex_enc = 0 if sex == "Female" else (1 if sex == "Male" else 2)
        
        # Simple hash encoding as a placeholder (Ensure this matches Colab logic)
        brand_enc = abs(hash(brand)) % 1000
        class_enc = abs(hash(p_class)) % 1000
        
        # FIX: The array below now contains exactly 9 features in the correct order.
        # Order: year, month, age, weight, sex, brand, class, n_drugs, n_react
        input_data = np.array([[
            year, 
            month, 
            age, 
            weight, 
            sex_enc, 
            brand_enc, 
            class_enc, 
            n_drugs, 
            n_react
        ]])
        
        try:
            # 4. Scale and Predict
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            proba = model.predict_proba(scaled_data)[0][1]
            
            # 5. Display Results
            st.subheader("Analysis Results")
            if prediction == 1:
                st.error(f"SERIOUS OUTCOME PREDICTED (Probability: {proba:.2%})")
                st.warning("Recommendation: Immediate medical review required.")
            else:
                st.success(f"NOT SERIOUS (Probability of seriousness: {proba:.2%})")
        except Exception as prediction_error:
            st.error(f"Prediction failed: {prediction_error}")
            st.info("Technical Note: The Scaler expects 9 features. Ensure 'quarter' is excluded.")