import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    # Loading the three pillars of your model
    model = joblib.load('fda_random_forest_model.pkl')
    scaler = joblib.load('fda_scaler.pkl')
    encoder = joblib.load('fda_encoder.pkl') 
    return model, scaler, encoder

model, scaler, encoder = None, None, None
try:
    model, scaler, encoder = load_assets()
    st.sidebar.success("✅ Model Assets Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading .pkl files: {e}")

# 2. UI HEADER
st.set_page_config(page_title="FDA Risk Predictor", page_icon="🛡️")
st.title("🛡️ FDA Adverse Event Risk Predictor")
st.markdown("""
This application uses a **Random Forest Classifier** trained on 528,000 FDA reports 
to identify high-risk medical outcomes based on patient complexity.
""")

# 3. SIDEBAR INPUTS (PHASE 7: DEPLOYMENT)
st.sidebar.header("📋 Patient Clinical Profile")
year = st.sidebar.slider("Report Year", 2015, 2030, 2026)
month = st.sidebar.slider("Report Month", 1, 12, 5)
age = st.sidebar.number_input("Patient Age", 0, 110, 65)
weight = st.sidebar.number_input("Weight (kg)", 0, 250, 70)

# Categorical Inputs (Processed via fda_encoder.pkl)
st.sidebar.subheader("Categorical Data")
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Unknown"])
brand = st.sidebar.text_input("Drug Brand", "ADVATE")
p_class = st.sidebar.text_input("Pharm Class", "Antihemophilic Factor [EPC]")

# Complexity Inputs (Top predictors from Phase 4)
st.sidebar.subheader("Complexity Factors")
n_drugs = st.sidebar.number_input("Number of Drugs Taken", 1, 50, 15)
n_react = st.sidebar.number_input("Number of Reactions", 1, 50, 10)

# Debug mode for technical demonstration
debug = st.sidebar.checkbox("Show Technical Debug Info")

# 4. PREDICTION LOGIC (PHASE 3 & 4 ALIGNMENT)
if st.button("Analyze Risk Profile"):
    if model and scaler and encoder:
        try:
            # A. ENCODING: Transforming text to the model's language
            cat_df = pd.DataFrame([[sex, brand, p_class]], 
                                  columns=['patient_sex', 'brand_name', 'pharm_class'])
            encoded_cats = encoder.transform(cat_df)
            
            # B. FEATURE ASSEMBLY: Exact 9 features in order
            input_data = np.array([[
                year, month, age, weight, 
                encoded_cats[0][0], encoded_cats[0][1], encoded_cats[0][2], 
                n_drugs, n_react
            ]])
            
            # C. SCALING: Using fda_scaler.pkl
            scaled_data = scaler.transform(input_data)
            
            # D. INFERENCE
            prediction = model.predict(scaled_data)[0]
            proba_serious = model.predict_proba(scaled_data)[0][1]

            if debug:
                st.write(f"**Encoded Feature Vector:** {input_data}")
                st.write(f"**Raw Model Output:** {prediction}")

            st.divider()
            
            # E. DYNAMIC OUTPUT (The Logic Fix)
            # This handles both numeric (1) and string ('Yes') labels
            if str(prediction) == '1' or str(prediction).lower() == 'yes':
                st.error("### ⚠️ HIGH RISK: SERIOUS OUTCOME PREDICTED")
                st.write("This report exhibits clinical patterns strongly correlated with hospitalization or life-threatening outcomes.")
                st.metric("Risk Confidence Score", f"{proba_serious:.2%}")
            else:
                st.success("### ✅ LOW RISK: NOT SERIOUS")
                st.write("The patient profile suggests a standard adverse event with a high probability of non-serious recovery.")
                st.metric("Probability of Seriousness", f"{proba_serious:.2%}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Note: Ensure 'Drug Brand' matches training data categories.")
    else:
        st.warning("Prediction unavailable. Ensure all .pkl files are in the directory.")

# 5. MODEL INSIGHTS (PHASE 6: VISUALIZATION)
st.divider()
st.header("📊 Model Decision Basis")
tab1, tab2 = st.tabs(["Feature Importance", "CRISP-DM Context"])

with tab1:
    st.subheader("What drives this result?")
    if model:
        importances = model.feature_importances_
        features = ['Year', 'Month', 'Age', 'Weight', 'Sex', 'Brand', 'Class', 'Drugs', 'Reactions']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances, y=features, palette='magma', ax=ax)
        ax.set_title("Impact Weight of Clinical Features")
        st.pyplot(fig)
        
        st.info("**Polypharmacy Alert:** The 'Number of Drugs' is the primary driver of risk in this model.")

with tab2:
    st.write("### Project Technical Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Preparation:**")
        st.write("- 528k FDA Records")
        st.write("- Standard Scaling Applied")
        st.write("- Ordinal Encoding for Text")
    with col2:
        st.write("**Model Evaluation:**")
        st.write("- Algorithm: Random Forest")
        st.write("- Recall Score: 74.86%")
        st.write("- Framework: CRISP-DM")