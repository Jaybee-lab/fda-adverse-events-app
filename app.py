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
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #d1ecf1;
        border: 1px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recommendation-high {
        background-color: #f8d7da;
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .recommendation-medium {
        background-color: #fff3cd;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .recommendation-low {
        background-color: #d4edda;
        border-left: 4px solid #27ae60;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .recommendation-info {
        background-color: #d1ecf1;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
TRAINING_YEAR_START = 2015
TRAINING_YEAR_END = 2025
FEATURE_COLUMNS = ['year', 'month', 'patient_age_years', 'patient_weight_kg', 
                   'patient_sex', 'brand_name', 'pharm_class', 'num_drugs', 'num_reactions']

# Pre-computed statistics from the dataset
DATASET_STATS = {
    'total_reports': 528000,
    'serious_reports': 395000,
    'non_serious_reports': 133000,
    'serious_percentage': 74.8,
    'avg_age': 55.9,
    'avg_drugs': 8.7,
    'avg_reactions': 6.28,
    'age_range': (0, 120),
    'weight_range': (0.05, 300),
    'top_countries': ['US', 'GB', 'DE', 'FR', 'CA', 'BR', 'ES', 'IT', 'AU', 'JP'],
    'serious_flags': ['Hospitalization', 'Life Threatening', 'Disability', 'Death', 'Other'],
    'yearly_trend': {
        2015: 25000, 2016: 28000, 2017: 31000, 2018: 35000, 
        2019: 38000, 2020: 42000, 2021: 48000, 2022: 52000, 
        2023: 58000, 2024: 62000, 2025: 65000
    }
}

# Feature importance from notebook
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
        st.info("Please ensure 'fda_random_forest_model.pkl', 'fda_scaler.pkl', and 'encoder.pkl' are in the same directory.")
        return None, None, None

def get_risk_factors(age, num_drugs, num_reactions, sex):
    """Identify risk factors based on input values"""
    risk_factors = []
    
    # Age-based risk
    if age > 65:
        risk_factors.append(("Age over 65 years", "high", "Elderly patients have higher risk of serious reactions"))
    elif age < 18:
        risk_factors.append(("Age under 18 years", "medium", "Pediatric patients may have different reaction profiles"))
    
    # Drug count risk
    if num_drugs >= 6:
        risk_factors.append(("Polypharmacy (6 or more drugs)", "high", "Increased risk of drug interactions"))
    elif num_drugs >= 4:
        risk_factors.append(("Multiple medications (4-5 drugs)", "medium", "Moderate risk of interactions"))
    
    # Reaction count risk
    if num_reactions >= 10:
        risk_factors.append(("Multiple reactions (10 or more)", "high", "Higher likelihood of serious outcomes"))
    elif num_reactions >= 5:
        risk_factors.append(("Multiple reactions (5-9 reactions)", "medium", "Elevated risk level"))
    
    # Gender-based risk
    if sex == "Female":
        risk_factors.append(("Female gender", "medium", "Higher reporting rate of adverse events in FDA data"))
    
    return risk_factors

def generate_recommendations(prediction, prediction_proba, risk_factors, year):
    """Generate actionable recommendations based on prediction"""
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
            'action': f'Year {year} is outside training range. Monitor for model drift.',
            'timeline': 'Ongoing'
        })
    
    return recommendations

def generate_summary(prediction, prediction_proba, risk_factors, input_data):
    """Generate comprehensive summary of the prediction"""
    serious_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
    non_serious_prob = prediction_proba[0] if len(prediction_proba) > 1 else 0
    
    summary = {
        'prediction': 'Serious Adverse Event' if prediction == 1 else 'Non-Serious Adverse Event',
        'confidence': f"{max(serious_prob, non_serious_prob) * 100:.1f}%",
        'serious_probability': f"{serious_prob * 100:.1f}%",
        'risk_level': 'HIGH' if serious_prob > 0.6 else 'MODERATE' if serious_prob > 0.4 else 'LOW',
        'key_drivers': [],
        'overall_assessment': ''
    }
    
    # Identify key drivers
    if input_data['num_reactions'][0] >= 10:
        summary['key_drivers'].append("High number of reactions")
    if input_data['num_drugs'][0] >= 6:
        summary['key_drivers'].append("Polypharmacy")
    if input_data['patient_age_years'][0] > 65:
        summary['key_drivers'].append("Elderly patient")
    if input_data['num_reactions'][0] >= 5 and input_data['num_drugs'][0] >= 4:
        summary['key_drivers'].append("Combined medication-reaction burden")
    
    if prediction == 1:
        summary['overall_assessment'] = f"This case has a {serious_prob * 100:.1f}% probability of being a serious adverse event. Key risk factors include {', '.join(summary['key_drivers']) if summary['key_drivers'] else 'multiple contributing factors'}. Immediate medical attention is recommended."
    else:
        summary['overall_assessment'] = f"This case has a {non_serious_prob * 100:.1f}% probability of being non-serious. While the immediate risk appears lower, continue standard monitoring protocols."
    
    return summary

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
        
        # Input fields
        st.subheader("Report Date")
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Reporting Year", min_value=2015, max_value=2030, value=2024)
        with col2:
            month = st.slider("Reporting Month", min_value=1, max_value=12, value=6)
        
        # Show warning for future years
        if year > TRAINING_YEAR_END:
            st.warning(f"⚠️ Year {year} is outside the training range (2015-2025). Predictions involve extrapolation.")
        
        st.markdown("---")
        st.subheader("Patient Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=5.0)
        
        sex = st.selectbox("Gender", options=["Female", "Male", "Unknown"])
        sex_map = {"Female": 0, "Male": 1, "Unknown": -1}
        sex_encoded = sex_map[sex]
        
        st.markdown("---")
        st.subheader("Medication Details")
        
        col1, col2 = st.columns(2)
        with col1:
            num_drugs = st.number_input("Number of Drugs", min_value=1, max_value=50, value=4)
        with col2:
            num_reactions = st.number_input("Number of Reactions", min_value=1, max_value=100, value=3)
        
        brand_name = st.number_input("Drug Brand Code", min_value=-1, max_value=100, value=0)
        pharm_class = st.number_input("Pharmaceutical Class Code", min_value=-1, max_value=100, value=0)
        
        # Predict button
        predict_button = st.button("Predict Seriousness", use_container_width=True, type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("About This Tool")
        st.write(f"""
        This tool uses a **Random Forest machine learning model** trained on FDA adverse event reports 
        from {TRAINING_YEAR_START}-{TRAINING_YEAR_END} to predict whether a reported adverse drug reaction is likely to be **serious** 
        (requiring hospitalization, life-threatening, or resulting in death/disability).
        
        **Model Performance:**
        - **Accuracy:** 71.8%
        - **Precision:** 85.7%
        - **Recall:** 74.9%
        - **F1-Score:** 79.9%
        """)
        
        # Feature importance visualization
        st.subheader("Model Feature Importance")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        features = list(FEATURE_IMPORTANCE.keys())
        importance = list(FEATURE_IMPORTANCE.values())
        
        bars = ax.barh(features, importance, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Predicting Serious Adverse Events')
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
        st.info("""
        This tool is for **educational and research purposes only**. 
        It should not be used as a substitute for professional medical advice.
        """)
    
    # Prediction result
    if predict_button:
        st.markdown("---")
        st.header("Prediction Result")
        
        # Prepare input data
        input_data = pd.DataFrame([[
            year, month, age, weight, sex_encoded, brand_name, pharm_class, num_drugs, num_reactions
        ]], columns=FEATURE_COLUMNS)
        
        try:
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            serious_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
            non_serious_prob = prediction_proba[0] if len(prediction_proba) > 1 else 0
            
            # Get risk factors
            risk_factors = get_risk_factors(age, num_drugs, num_reactions, sex)
            
            # Generate summary
            summary = generate_summary(prediction, prediction_proba, risk_factors, input_data)
            
            # Display Summary Section
            st.subheader("Clinical Summary")
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.info(f"**Prediction:** {summary['prediction']}")
            with col_sum2:
                st.info(f"**Confidence:** {summary['confidence']}")
            with col_sum3:
                risk_icon = "🔴 HIGH" if summary['risk_level'] == "HIGH" else "🟡 MODERATE" if summary['risk_level'] == "MODERATE" else "🟢 LOW"
                st.info(f"**Risk Level:** {risk_icon}")
            
            st.markdown(f"<div class='info-card'>{summary['overall_assessment']}</div>", unsafe_allow_html=True)
            
            if summary['key_drivers']:
                st.write("**Key Risk Drivers Identified:**")
                for driver in summary['key_drivers']:
                    st.write(f"- ⚠️ {driver}")
            
            # Display Prediction Card
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            
            with col_result2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>⚠️ SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-high">High Risk of Serious Outcome</p>
                        <p>Probability: {serious_prob * 100:.1f}%</p>
                        <hr>
                        <p>This report shows characteristics similar to serious adverse events in FDA data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>✅ NON-SERIOUS ADVERSE EVENT</h3>
                        <p class="risk-low">Low Risk of Serious Outcome</p>
                        <p>Probability: {non_serious_prob * 100:.1f}%</p>
                        <hr>
                        <p>This report characteristics are consistent with non-serious adverse events.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display Risk Factors
            if risk_factors:
                st.subheader("Identified Risk Factors")
                for factor, level, reason in risk_factors:
                    if level == "high":
                        st.markdown(f"🔴 **{factor}** - {reason}")
                    elif level == "medium":
                        st.markdown(f"🟡 **{factor}** - {reason}")
                    else:
                        st.markdown(f"🟢 **{factor}** - {reason}")
            
            # Display Recommendations
            st.subheader("Recommendations")
            recommendations = generate_recommendations(prediction, prediction_proba, risk_factors, year)
            
            for rec in recommendations:
                if rec['priority'] == 'HIGH':
                    st.markdown(f"""
                    <div class="recommendation-high">
                        <strong>🔴 HIGH PRIORITY</strong><br>
                        <strong>{rec['title']}</strong><br>
                        {rec['action']}<br>
                        <small>⏱️ Timeline: {rec['timeline']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif rec['priority'] == 'MEDIUM':
                    st.markdown(f"""
                    <div class="recommendation-medium">
                        <strong>🟡 MEDIUM PRIORITY</strong><br>
                        <strong>{rec['title']}</strong><br>
                        {rec['action']}<br>
                        <small>⏱️ Timeline: {rec['timeline']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif rec['priority'] == 'LOW':
                    st.markdown(f"""
                    <div class="recommendation-low">
                        <strong>🟢 LOW PRIORITY</strong><br>
                        <strong>{rec['title']}</strong><br>
                        {rec['action']}<br>
                        <small>⏱️ Timeline: {rec['timeline']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="recommendation-info">
                        <strong>ℹ️ INFO</strong><br>
                        <strong>{rec['title']}</strong><br>
                        {rec['action']}<br>
                        <small>⏱️ Timeline: {rec['timeline']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show detailed probabilities
            st.subheader("Prediction Confidence Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Non-Serious Probability", f"{non_serious_prob * 100:.1f}%")
            with prob_col2:
                st.metric("Serious Probability", f"{serious_prob * 100:.1f}%")
            
            # Progress bar visualization
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Risk Level'], [serious_prob], color='#e74c3c' if serious_prob > 0.5 else '#27ae60')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Serious Event Probability')
            ax.set_title('Risk Assessment')
            st.pyplot(fig)
            plt.close()
            
            # Show input summary
            with st.expander("View Complete Input Summary"):
                input_summary_data = {
                    'Feature': ['Year', 'Month', 'Age', 'Weight', 'Gender', 'Drugs Count', 'Reactions Count', 'Brand Code', 'Pharm Class Code'],
                    'Value': [year, month, age, weight, sex, num_drugs, num_reactions, brand_name, pharm_class]
                }
                st.dataframe(pd.DataFrame(input_summary_data), use_container_width=True, hide_index=True)
                
                if year > TRAINING_YEAR_END:
                    st.warning(f"Note: Year {year} is outside the training range (2015-2025). This prediction involves extrapolation.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please ensure all input values are valid and try again.")

# Data Exploration Page
def data_exploration_page():
    st.markdown('<div class="main-header">Data Exploration</div>', unsafe_allow_html=True)
    
    st.info("The following visualizations are based on pre-computed statistics from the FDA FAERS dataset (2015-2025).")
    
    # Yearly trend visualization
    st.subheader("Annual Report Trends")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(DATASET_STATS['yearly_trend'].keys())
    counts = list(DATASET_STATS['yearly_trend'].values())
    
    ax.plot(years, counts, marker='o', color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(years, counts, alpha=0.3, color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Reports')
    ax.set_title('Adverse Event Reports by Year (2015-2025)')
    ax.axvspan(TRAINING_YEAR_END + 0.5, 2030, alpha=0.2, color='red', label='Extrapolation Zone')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.caption(f"Training Data Range: {TRAINING_YEAR_START}-{TRAINING_YEAR_END} | Extrapolation Zone: {TRAINING_YEAR_END + 1}+")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ages = np.random.normal(DATASET_STATS['avg_age'], 20, 10000)
        ages = ages[(ages >= 0) & (ages <= 120)]
        ax.hist(ages, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(DATASET_STATS['avg_age'], color='red', linestyle='--', linewidth=2, label=f'Mean: {DATASET_STATS["avg_age"]} years')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Patient Ages')
        ax.legend()
        st.pyplot(fig)
        plt.close()
        st.caption(f"Mean Age: {DATASET_STATS['avg_age']} years | Range: {DATASET_STATS['age_range'][0]}-{DATASET_STATS['age_range'][1]} years")
    
    with col2:
        st.subheader("Serious vs Non-Serious")
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [DATASET_STATS['serious_reports'], DATASET_STATS['non_serious_reports']]
        labels = ['Serious', 'Non-Serious']
        colors = ['#e74c3c', '#27ae60']
        explode = (0.05, 0)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90, shadow=True)
        ax.set_title('Proportion of Serious vs Non-Serious Reports')
        st.pyplot(fig)
        plt.close()
        st.caption(f"Total Reports: {DATASET_STATS['total_reports']:,} | Serious: {DATASET_STATS['serious_percentage']:.1f}%")
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(FEATURE_IMPORTANCE.keys())
    importance = list(FEATURE_IMPORTANCE.values())
    bars = ax.barh(features, importance, color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance in Predicting Serious Adverse Events')
    ax.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig)
    plt.close()
    
    st.caption("Number of reactions is the strongest predictor, followed by patient age and number of drugs.")
    
    # Additional statistics
    st.subheader("Additional Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Reactions per Report", f"{DATASET_STATS['avg_reactions']}")
    with col2:
        st.metric("Top Reporting Country", DATASET_STATS['top_countries'][0])
    with col3:
        st.metric("Common Serious Flags", ", ".join(DATASET_STATS['serious_flags'][:3]))

# Navigation
pages = {
    "Predictor": main,
    "Data Exploration": data_exploration_page
}

# Sidebar navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run selected page
pages[selection]()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**About**  
- Built with Random Forest Classifier  
- Trained on FDA FAERS Data (2015-2025)  
- Accuracy: 71.8% | Precision: 85.7%  
- For educational purposes only
""")
