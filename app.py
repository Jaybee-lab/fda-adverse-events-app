import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Assets
@st.cache_resource
def load_assets():
    # Load the three essential pkl files
    model = joblib.load('fda_random_forest_model.pkl')
    scaler = joblib.load('fda_scaler.pkl')
    encoder = joblib.load('fda_encoder.pkl') 
    return model, scaler, encoder

model, scaler, encoder = None, None, None
try:
    model, scaler, encoder = load_assets()
    st.sidebar.success("✅ Model Assets Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model files: {e}")

# 2. UI Header
st.title("🛡️ FDA Adverse Event Risk Predictor")
st.markdown("This tool uses machine learning to predict if a drug reaction is likely to be **Serious** based on patient demographics and clinical complexity.")

# 3. Sidebar Inputs
st.sidebar.header("Patient & Clinical Data")
year = st.sidebar.slider("Year", 2015, 2030, 2026)
month = st.sidebar.slider("Month", 1, 12, 5)
age = st.sidebar.number_input("Patient Age", 0, 110, 50)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)

# Categorical Inputs
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "Generic")
p_class = st.sidebar.text_input("Pharm Class", "Unknown")

# Clinical Complexity Inputs
n_drugs = st.sidebar.number_input("Number of Drugs Taken", 1, 50, 2)
n_react = st.sidebar.number_input("Number of Reactions", 1, 50, 1)

# Debug mode for presentation
debug = st.sidebar.checkbox("Show Debug Info")

# 4. Prediction Logic
if st.button("Run Risk Analysis"):
    if model and scaler and encoder:
        try:
            # A. ENCODING CATEGORICAL DATA
            # We use a DataFrame so the encoder recognizes the column names
            cat_df = pd.DataFrame([[sex, brand, p_class]], 
                                  columns=['patient_sex', 'brand_name', 'pharm_class'])
            
            # Transform categorical data using your fda_encoder.pkl
            encoded_cats = encoder.transform(cat_df)
            sex_enc = encoded_cats[0][0]
            brand_enc = encoded_cats[0][1]
            class_enc = encoded_cats[0][2]

            # B. CONSTRUCT INPUT ARRAY
            # Exact 9 features in the order your model expects
            input_data = np.array([[year, month, age, weight, sex_enc, brand_enc, class_enc, n_drugs, n_react]])
            
            # C. SCALING
            scaled_data = scaler.transform(input_data)
            
            # D. PREDICTION
            prediction = model.predict(scaled_data)[0]
            proba_serious = model.predict_proba(scaled_data)[0][1] # Probability of 'Serious' (1)

            if debug:
                st.write(f"DEBUG - Encoded Values: Sex={sex_enc}, Brand={brand_enc}, Class={class_enc}")
                st.write(f"DEBUG - Raw Prediction: {prediction}")

            st.divider()
            
            # E. DISPLAY RESULTS
            # Logic handles both numeric (1) and string ('Yes') outputs
            if str(prediction) == '1' or str(prediction).lower() == 'yes':
                st.error("### ⚠️ HIGH RISK: SERIOUS OUTCOME PREDICTED")
                st.write("The model identifies clinical patterns often associated with serious medical events.")
                st.metric("Risk Confidence", f"{proba_serious:.2%}")
            else:
                st.success("### ✅ LOW RISK: NOT SERIOUS")
                st.write("The model identifies characteristics typically associated with non-serious outcomes.")
                st.metric("Probability of Seriousness", f"{proba_serious:.2%}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure the Drug Brand and Pharm Class exist in the training data or check for typos.")
    else:
        st.warning("Model assets are missing. Please upload your .pkl files.")

# 5. Visualizations Section
st.divider()
st.header("📊 Model Insights")
tab1, tab2 = st.tabs(["Feature Importance", "Project Summary"])

with tab1:
    st.subheader("What drives this prediction?")
    if model:
        importances = model.feature_importances_
        feature_names = ['Year', 'Month', 'Age', 'Weight', 'Sex', 'Brand', 'Class', 'Drugs', 'Reactions']
        
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, palette='viridis', ax=ax)
        ax.set_title("Top Risk Predictors")
        st.pyplot(fig)
        
        st.info("The model relies most heavily on **Number of Drugs** (Polypharmacy) and **Number of Reactions**.")

with tab2:
    st.subheader("Technical Context")
    st.write("- **Dataset:** 528,000 FDA Adverse Event Reports (2015-2025)")
    st.write("- **Methodology:** CRISP-DM Framework")
    st.write("- **Model Performance:** 74.86% Recall for Serious Outcomes")