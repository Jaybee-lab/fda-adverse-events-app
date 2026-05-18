import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FDA Adverse Event Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-high {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load pre-trained models and preprocessors"""
    try:
        model = joblib.load('fda_random_forest_model.pkl')
        scaler = joblib.load('fda_scaler.pkl')
        encoder = joblib.load('fda_encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure 'fda_random_forest_model.pkl', 'fda_scaler.pkl', and 'encoder.pkl' are in the same directory.")
        return None, None, None

# Feature columns (must match what the model was trained on)
FEATURE_COLUMNS = ['year', 'month', 'patient_age_years', 'patient_weight_kg', 
                   'patient_sex', 'brand_name', 'pharm_class', 'num_drugs', 'num_reactions']

def main():
    # Header
    st.markdown('<div class="main-header">🏥 FDA Adverse Event Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict whether an adverse drug reaction will be serious using machine learning</div>', unsafe_allow_html=True)
    
    # Load models
    model, scaler, encoder = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Patient & Report Information")
        st.markdown("---")
        
        # Input fields
        year = st.number_input("Reporting Year", min_value=2015, max_value=2026, value=2024)
        month = st.slider("Reporting Month", min_value=1, max_value=12, value=6)
        
        st.markdown("---")
        st.subheader("👤 Patient Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=5.0)
        
        sex = st.selectbox("Gender", options=["Female", "Male", "Unknown"])
        sex_map = {"Female": 0, "Male": 1, "Unknown": -1}
        sex_encoded = sex_map[sex]
        
        st.markdown("---")
        st.subheader("💊 Medication Details")
        
        num_drugs = st.number_input("Number of Drugs", min_value=1, max_value=50, value=4)
        num_reactions = st.number_input("Number of Reactions", min_value=1, max_value=100, value=3)
        
        # For encoded features, use typical values
        brand_name = st.number_input("Drug Brand Code", min_value=-1, max_value=100, value=0, help="Encoded brand identifier")
        pharm_class = st.number_input("Pharmaceutical Class Code", min_value=-1, max_value=100, value=0, help="Encoded pharmaceutical class")
        
        # Predict button
        predict_button = st.button("🔮 Predict Seriousness", use_container_width=True, type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 About This Tool")
        st.write("""
        This tool uses a **Random Forest machine learning model** trained on FDA adverse event reports 
        from 2015-2026 to predict whether a reported adverse drug reaction is likely to be **serious** 
        (requiring hospitalization, life-threatening, or resulting in death/disability).
        
        **Key model performance metrics:**
        - **Accuracy:** 71.8%
        - **Precision:** 85.7%
        - **Recall:** 74.9%
        - **F1-Score:** 79.9%
        """)
        
        # Feature importance visualization
        st.subheader("🔍 Model Feature Importance")
        feature_importance_data = {
            'Feature': ['num_reactions', 'patient_age_years', 'num_drugs', 'pharm_class', 
                       'brand_name', 'patient_sex', 'patient_weight_kg', 'month', 'year'],
            'Importance': [0.32, 0.18, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02]
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        importance_df = pd.DataFrame(feature_importance_data).sort_values('Importance', ascending=True)
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Predicting Serious Adverse Events')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📈 Quick Stats")
        st.metric("Total Reports Analyzed", "528,000")
        st.metric("Serious Reports", "395,000 (74.8%)")
        st.metric("Non-Serious Reports", "133,000 (25.2%)")
        st.metric("Average Patient Age", "55.9 years")
        st.metric("Average Drugs per Report", "8.7")
        
        st.markdown("---")
        st.subheader("⚠️ Disclaimer")
        st.info("""
        This tool is for **educational and research purposes only**. 
        It should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always consult with qualified healthcare 
        professionals for medical decisions.
        """)
    
    # Prediction result
    if predict_button:
        st.markdown("---")
        st.header("🔮 Prediction Result")
        
        # Prepare input data with correct column order
        input_data = pd.DataFrame([[
            year, month, age, weight, sex_encoded, brand_name, pharm_class, num_drugs, num_reactions
        ]], columns=FEATURE_COLUMNS)
        
        try:
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display result
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            
            with col_result2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>⚠️ PREDICTION: SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-high">High Risk of Serious Outcome</p>
                        <p>Probability: {prediction_proba[1] * 100:.1f}%</p>
                        <hr>
                        <p style="color: #e74c3c;">This report shows characteristics similar to serious adverse events in FDA data.</p>
                        <p><strong>Recommended Action:</strong> Medical review recommended. Monitor patient closely.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>✅ PREDICTION: NON-SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-low">Low Risk of Serious Outcome</p>
                        <p>Probability: {prediction_proba[0] * 100:.1f}%</p>
                        <hr>
                        <p style="color: #27ae60;">This report characteristics are consistent with non-serious adverse events.</p>
                        <p><strong>Recommended Action:</strong> Continue standard monitoring protocols.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show detailed probabilities
            st.subheader("📊 Prediction Confidence Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Non-Serious Probability", f"{prediction_proba[0] * 100:.1f}%")
            with prob_col2:
                st.metric("Serious Probability", f"{prediction_proba[1] * 100:.1f}%")
            
            # Progress bar visualization
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Risk Level'], [prediction_proba[1]], color='#e74c3c' if prediction_proba[1] > 0.5 else '#27ae60')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Serious Event Probability')
            ax.set_title('Risk Assessment')
            st.pyplot(fig)
            plt.close()
            
            # Show input summary
            with st.expander("View Input Summary"):
                input_summary = {
                    'Feature': ['Year', 'Month', 'Age', 'Weight', 'Gender', 'Drugs Count', 'Reactions Count'],
                    'Value': [year, month, age, weight, sex, num_drugs, num_reactions]
                }
                st.dataframe(pd.DataFrame(input_summary), use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please ensure all input values are valid and try again.")

# Additional page for data exploration
def data_exploration_page():
    st.markdown('<div class="main-header">📈 Data Exploration</div>', unsafe_allow_html=True)
    
    # Load a sample of the data for visualization
    @st.cache_data
    def load_sample_data():
        try:
            df = pd.read_csv('fda_adverse_events_2015_2026_CLEAN.csv', nrows=10000)
            return df
        except FileNotFoundError:
            st.warning("Data file 'fda_adverse_events_2015_2026_CLEAN.csv' not found.")
            st.info("If you want data exploration features, please ensure the CSV file is in the same directory.")
            return None
        except Exception as e:
            st.warning(f"Error loading data: {e}")
            return None
    
    df_sample = load_sample_data()
    
    if df_sample is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_sample['patient_age_years'].dropna().hist(bins=30, color='steelblue', edgecolor='black', ax=ax)
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Patient Ages')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Serious vs Non-Serious")
            fig, ax = plt.subplots(figsize=(8, 5))
            # Handle both string and numeric serious values
            if df_sample['serious'].dtype == 'object':
                serious_counts = df_sample['serious'].value_counts()
            else:
                serious_counts = df_sample['serious'].map({1: 'Yes', 0: 'No'}).value_counts()
            
            colors = ['#e74c3c', '#27ae60']
            ax.pie(serious_counts.values, labels=serious_counts.index, autopct='%1.1f%%', colors=colors[:len(serious_counts)], startangle=90)
            ax.set_title('Proportion of Serious vs Non-Serious Reports')
            st.pyplot(fig)
            plt.close()
        
        st.subheader("Drug Count Distribution by Seriousness")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_clean = df_sample[df_sample['num_drugs'] <= 50].copy()
        
        # Handle serious column for boxplot
        if df_clean['serious'].dtype == 'object':
            df_clean['serious_label'] = df_clean['serious']
        else:
            df_clean['serious_label'] = df_clean['serious'].map({1: 'Serious', 0: 'Non-Serious'})
        
        df_clean.boxplot(column='num_drugs', by='serious_label', ax=ax)
        ax.set_title('Number of Drugs by Seriousness')
        ax.set_xlabel('Seriousness')
        ax.set_ylabel('Number of Drugs')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Upload the CSV file to enable data exploration features.")

# Navigation
pages = {
    "🏠 Predictor": main,
    "📊 Data Exploration": data_exploration_page
}

# Sidebar navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run selected page
pages[selection]()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**  
Built with Random Forest Classifier  
Trained on FDA FAERS Data (2015-2026)  
For educational purposes only
""")
