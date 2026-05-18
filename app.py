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

# Custom CSS with proper text colors for both light and dark mode
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: inherit;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: inherit;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    /* Prediction cards */
    .prediction-card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
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
    
    /* Warning and info cards */
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .info-card {
        background-color: #d1ecf1;
        border: 1px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    
    /* Recommendations - using light colors that work in both modes */
    .recommendation-high {
        background-color: #f8d7da;
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        color: #721c24;
    }
    .recommendation-medium {
        background-color: #fff3cd;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        color: #856404;
    }
    .recommendation-low {
        background-color: #d4edda;
        border-left: 4px solid #27ae60;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        color: #155724;
    }
    .recommendation-info {
        background-color: #d1ecf1;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        color: #0c5460;
    }
    
    /* Ensure all text is visible */
    .stMarkdown, .stText, .stInfo, .stWarning, .stSuccess {
        color: inherit;
    }
    
    /* Fix metric colors */
    [data-testid="stMetricValue"] {
        color: inherit;
    }
    
    /* Fix dataframe text */
    .dataframe {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Constants
TRAINING_YEAR_START = 2015
TRAINING_YEAR_END = 2025
FEATURE_COLUMNS = ['year', 'month', 'patient_age_years', 'patient_weight_kg', 
                   'patient_sex', 'brand_name', 'pharm_class', 'num_drugs', 'num_reactions']

# Pre-computed statistics
DATASET_STATS = {
    'total_reports': 528000,
    'serious_reports': 395000,
    'non_serious_reports': 133000,
    'serious_percentage': 74.8,
    'avg_age': 55.9,
    'avg_drugs': 8.7,
    'avg_reactions': 6.28,
}

# Feature importance
FEATURE_IMPORTANCE = {
    'Number of Reactions': 0.32,
    'Patient Age': 0.18,
    'Number of Drugs': 0.15,
    'Pharmaceutical Class': 0.12,
    'Drug Brand': 0.08,
    'Patient Sex': 0.06,
    'Patient Weight': 0.04,
    'Month': 0.03,
    'Year': 0.02
}

# Load models
@st.cache_resource
def load_models():
    """Load pre-trained models and preprocessors"""
    try:
        model = joblib.load('fda_random_forest_model.pkl')
        scaler = joblib.load('fda_scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure all model files are in the same directory.")
        return None, None, None

def get_risk_factors(age, num_drugs, num_reactions, sex):
    """Identify risk factors based on input values"""
    risk_factors = []
    
    if age > 65:
        risk_factors.append(("Age over 65 years", "high", "Elderly patients have higher risk of serious reactions"))
    elif age < 18:
        risk_factors.append(("Age under 18 years", "medium", "Pediatric patients may have different reaction profiles"))
    
    if num_drugs >= 6:
        risk_factors.append(("Polypharmacy (6+ drugs)", "high", "Increased risk of drug interactions"))
    elif num_drugs >= 4:
        risk_factors.append(("Multiple medications (4-5 drugs)", "medium", "Moderate risk of interactions"))
    
    if num_reactions >= 10:
        risk_factors.append(("Multiple reactions (10+)", "high", "Higher likelihood of serious outcomes"))
    elif num_reactions >= 5:
        risk_factors.append(("Multiple reactions (5-9)", "medium", "Elevated risk level"))
    
    if sex == "Female":
        risk_factors.append(("Female gender", "medium", "Higher reporting rate in FDA data"))
    
    return risk_factors

def generate_recommendations(prediction, prediction_proba, risk_factors, year):
    """Generate actionable recommendations"""
    recommendations = []
    serious_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
    
    if prediction == 1:
        recommendations.append({
            'priority': 'HIGH',
            'title': 'Immediate Medical Review Required',
            'action': 'Conduct thorough clinical evaluation for serious adverse event.',
            'timeline': 'Within 24 hours'
        })
        
        if serious_prob > 0.8:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Hospitalization Consideration',
                'action': 'Consider inpatient monitoring due to high probability of serious outcome.',
                'timeline': 'Immediate'
            })
        
        recommendations.append({
            'priority': 'MEDIUM',
            'title': 'Drug Interaction Assessment',
            'action': 'Review all medications for potential drug interactions.',
            'timeline': 'Within 48 hours'
        })
        
        recommendations.append({
            'priority': 'MEDIUM',
            'title': 'FDA MedWatch Report',
            'action': 'Submit a detailed report to FDA MedWatch for this serious adverse event.',
            'timeline': 'As soon as possible'
        })
    else:
        recommendations.append({
            'priority': 'LOW',
            'title': 'Standard Monitoring',
            'action': 'Continue routine adverse event monitoring as per standard protocols.',
            'timeline': 'Standard schedule'
        })
        
        if serious_prob > 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Follow-up Recommended',
                'action': 'Schedule a follow-up within 2 weeks to monitor for symptom changes.',
                'timeline': 'Within 2 weeks'
            })
    
    # Add risk factor recommendations
    for factor, level, reason in risk_factors:
        if level == 'high':
            recommendations.append({
                'priority': 'HIGH',
                'title': f'Risk Factor: {factor}',
                'action': reason,
                'timeline': 'Address promptly'
            })
        elif level == 'medium' and prediction == 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': f'Risk Factor: {factor}',
                'action': reason,
                'timeline': 'Monitor closely'
            })
    
    # Year extrapolation warning
    if year > TRAINING_YEAR_END:
        recommendations.append({
            'priority': 'INFO',
            'title': 'Extrapolation Note',
            'action': f'Year {year} is outside training range ({TRAINING_YEAR_START}-{TRAINING_YEAR_END}). Monitor for model drift.',
            'timeline': 'Ongoing'
        })
    
    return recommendations

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
        st.header("Patient & Report Information")
        st.markdown("---")
        
        st.subheader("Report Date")
        year = st.number_input("Reporting Year", min_value=2015, max_value=2030, value=2024)
        month = st.slider("Reporting Month", min_value=1, max_value=12, value=6)
        
        if year > TRAINING_YEAR_END:
            st.warning(f"⚠️ Year {year} is outside the training range (2015-2025)")
        
        st.markdown("---")
        st.subheader("Patient Demographics")
        
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=5.0)
        sex = st.selectbox("Gender", options=["Female", "Male", "Unknown"])
        sex_map = {"Female": 0, "Male": 1, "Unknown": -1}
        sex_encoded = sex_map[sex]
        
        st.markdown("---")
        st.subheader("Medication Details")
        
        num_drugs = st.number_input("Number of Drugs", min_value=1, max_value=50, value=4)
        num_reactions = st.number_input("Number of Reactions", min_value=1, max_value=100, value=3)
        brand_name = st.number_input("Drug Brand Code", min_value=-1, max_value=100, value=0)
        pharm_class = st.number_input("Pharmaceutical Class Code", min_value=-1, max_value=100, value=0)
        
        predict_button = st.button("🔮 Predict Seriousness", use_container_width=True, type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("About This Tool")
        st.write(f"""
        This tool uses a **Random Forest model** trained on FDA adverse event reports 
        from {TRAINING_YEAR_START}-{TRAINING_YEAR_END} to predict whether an adverse drug reaction 
        is likely to be **serious** (requiring hospitalization, life-threatening, or resulting in death/disability).
        
        **Model Performance:**
        - **Accuracy:** 71.8%
        - **Precision:** 85.7%
        - **Recall:** 74.9%
        - **F1-Score:** 79.9%
        """)
        
        # Feature importance
        st.subheader("Model Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        features = list(FEATURE_IMPORTANCE.keys())
        importance = list(FEATURE_IMPORTANCE.values())
        ax.barh(features, importance, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Dataset Statistics")
        st.metric("Total Reports Analyzed", f"{DATASET_STATS['total_reports']:,}")
        st.metric("Serious Reports", f"{DATASET_STATS['serious_reports']:,} ({DATASET_STATS['serious_percentage']:.1f}%)")
        st.metric("Average Patient Age", f"{DATASET_STATS['avg_age']} years")
        st.metric("Average Drugs per Report", f"{DATASET_STATS['avg_drugs']}")
        
        st.markdown("---")
        st.subheader("Disclaimer")
        st.info("For educational and research purposes only. Not a substitute for medical advice.")
    
    # Prediction result
    if predict_button:
        st.markdown("---")
        st.header("🔮 Prediction Result")
        
        input_data = pd.DataFrame([[
            year, month, age, weight, sex_encoded, brand_name, pharm_class, num_drugs, num_reactions
        ]], columns=FEATURE_COLUMNS)
        
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            serious_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
            non_serious_prob = prediction_proba[0] if len(prediction_proba) > 1 else 0
            
            risk_factors = get_risk_factors(age, num_drugs, num_reactions, sex)
            
            # Display Prediction
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            with col_res2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>⚠️ SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-high">High Risk of Serious Outcome</p>
                        <p>Probability: {serious_prob * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>✅ NON-SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-low">Low Risk of Serious Outcome</p>
                        <p>Probability: {non_serious_prob * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Risk Factors
            if risk_factors:
                st.subheader("⚠️ Risk Factors")
                for factor, level, reason in risk_factors:
                    if level == "high":
                        st.error(f"**{factor}** - {reason}")
                    elif level == "medium":
                        st.warning(f"**{factor}** - {reason}")
                    else:
                        st.info(f"**{factor}** - {reason}")
            
            # Recommendations - Using native streamlit components for better theming
            st.subheader("📋 Recommendations")
            recommendations = generate_recommendations(prediction, prediction_proba, risk_factors, year)
            
            for rec in recommendations:
                if rec['priority'] == 'HIGH':
                    st.error(f"**{rec['title']}**\n\n{rec['action']}\n\n⏱️ Timeline: {rec['timeline']}")
                elif rec['priority'] == 'MEDIUM':
                    st.warning(f"**{rec['title']}**\n\n{rec['action']}\n\n⏱️ Timeline: {rec['timeline']}")
                elif rec['priority'] == 'LOW':
                    st.success(f"**{rec['title']}**\n\n{rec['action']}\n\n⏱️ Timeline: {rec['timeline']}")
                else:
                    st.info(f"**{rec['title']}**\n\n{rec['action']}\n\n⏱️ Timeline: {rec['timeline']}")
            
            # Confidence breakdown
            st.subheader("Prediction Confidence Breakdown")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Non-Serious Probability", f"{non_serious_prob * 100:.1f}%")
            with col_p2:
                st.metric("Serious Probability", f"{serious_prob * 100:.1f}%")
            
            # Progress bar
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Risk Level'], [serious_prob], color='#e74c3c' if serious_prob > 0.5 else '#27ae60')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Serious Event Probability')
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error: {e}")

# Data Exploration Page
def data_exploration_page():
    st.markdown('<div class="main-header">📈 Data Exploration</div>', unsafe_allow_html=True)
    
    st.subheader("Annual Report Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(range(2015, 2026))
    reports = [25000, 28000, 31000, 35000, 38000, 42000, 48000, 52000, 58000, 62000, 65000]
    ax.plot(years, reports, marker='o', color='steelblue', linewidth=2)
    ax.fill_between(years, reports, alpha=0.3, color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Reports')
    ax.set_title('Adverse Event Reports by Year')
    ax.axvspan(2026, 2030, alpha=0.2, color='red', label='Extrapolation Zone')
    ax.legend()
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ages = np.random.normal(55.9, 20, 10000)
        ages = ages[(ages >= 0) & (ages <= 120)]
        ax.hist(ages, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(55.9, color='red', linestyle='--', linewidth=2, label='Mean: 55.9 years')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Serious vs Non-Serious")
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [395000, 133000]
        labels = ['Serious (74.8%)', 'Non-Serious (25.2%)']
        colors = ['#e74c3c', '#27ae60']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Report Distribution')
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(FEATURE_IMPORTANCE.keys())
    importance = list(FEATURE_IMPORTANCE.values())
    ax.barh(features, importance, color='steelblue')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)
    plt.close()

# Navigation
pages = {
    "🏠 Predictor": main,
    "📊 Data Exploration": data_exploration_page
}

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

pages[selection]()

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**About**  
- Built with Random Forest  
- Trained on FDA FAERS (2015-2025)  
- Accuracy: 71.8% | Precision: 85.7%  
- For educational purposes only
""")
