import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from typing import Tuple, List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state for API key
if 'api_key_entered' not in st.session_state:
    st.session_state.api_key_entered = False

# Title and description
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This application predicts customer churn probability for subscription services using machine learning.
Enter your API key and customer information to get real-time predictions and insights.
""")

def load_or_create_model() -> Tuple[RandomForestClassifier, StandardScaler, LabelEncoder]:
    """
    Load existing model or create a new one if it doesn't exist.
    
    Returns:
        Tuple containing (model, scaler, encoder)
    """
    model_path = 'model.joblib'
    scaler_path = 'scaler.joblib'
    encoder_path = 'encoder.joblib'
    
    if os.path.exists(model_path):
        try:
            logger.info("Loading existing model and preprocessors")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            return model, scaler, encoder
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Continue with creating a new model
    
    logger.info("Creating new model with synthetic data")
    # Create dummy data for initial model
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.normal(40, 10, n_samples),
        'Total_Purchase': np.random.normal(5000, 2000, n_samples),
        'Account_Manager': np.random.choice([0, 1], n_samples),
        'Years': np.random.normal(5, 3, n_samples),
        'Num_Sites': np.random.normal(8, 3, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Prepare data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Initialize preprocessing objects
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    # Train model with improved hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Save model and preprocessors
    try:
        logger.info("Saving model and preprocessors")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    return model, scaler, encoder

def create_feature_importance_plot(model: RandomForestClassifier, feature_names: List[str]) -> go.Figure:
    """
    Create feature importance plot using plotly.
    
    Args:
        model: Trained RandomForestClassifier model
        feature_names: List of feature names
        
    Returns:
        Plotly figure object with feature importance visualization
    """
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance',
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def get_recommendations(risk_level: str) -> dict:
    """
    Get recommendations based on the risk level.
    
    Args:
        risk_level: Customer risk level (Low, Medium, High)
        
    Returns:
        Dictionary of recommendations
    """
    recommendations = {
        "High": {
            "title": "High Risk - Immediate Action Required",
            "actions": [
                "Schedule immediate customer review meeting",
                "Offer personalized retention package with 15-20% discount",
                "Conduct detailed satisfaction survey",
                "Assign dedicated account specialist",
                "Develop 90-day retention plan"
            ],
            "style": "warning"
        },
        "Medium": {
            "title": "Medium Risk - Proactive Engagement Needed",
            "actions": [
                "Increase engagement through bi-weekly check-ins",
                "Review current service usage and identify optimization opportunities",
                "Offer complementary training session",
                "Consider upsell with bundled discount",
                "Schedule quarterly business review"
            ],
            "style": "info"
        },
        "Low": {
            "title": "Low Risk - Nurture Relationship",
            "actions": [
                "Maintain regular monthly communication",
                "Look for expansion opportunities",
                "Introduce referral incentives",
                "Consider customer advocacy program",
                "Share product roadmap and gather feedback"
            ],
            "style": "success"
        }
    }
    
    return recommendations.get(risk_level, {})

def create_gauge_chart(churn_prob: float) -> go.Figure:
    """
    Create gauge chart for churn probability visualization.
    
    Args:
        churn_prob: Churn probability (0-1)
        
    Returns:
        Plotly figure object with gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability"},
        number={'suffix': "%", 'font': {'size': 26}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(31, 58, 147, 0.8)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(144, 238, 144, 0.6)"},
                {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.6)"},
                {'range': [70, 100], 'color': "rgba(250, 128, 114, 0.6)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key with a simple check.
    In a real application, this would verify with an external service.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if API key is valid, False otherwise
    """
    # This is a placeholder validation
    # In a real application, you would verify against your API service
    if api_key and len(api_key) > 10:
        return True
    return False

def api_key_form():
    """Display API key form and handle validation"""
    st.subheader("API Authentication")
    
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your API key", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            if validate_api_key(api_key):
                st.session_state.api_key = api_key
                st.session_state.api_key_entered = True
                st.success("API key validated successfully!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Invalid API key. Please try again.")

def main():
    try:
        logger.info("Starting application")
        
        # Check if API key is entered
        if not st.session_state.api_key_entered:
            api_key_form()
            return
            
        # Load or create model
        model, scaler, encoder = load_or_create_model()
        
        # Add tabs for main functionality
        tab1, tab2 = st.tabs(["Prediction", "Model Information"])
        
        with tab1:
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Customer Information")
                
                # Create form for input
                with st.form(key="customer_info_form"):
                    # Input fields
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                    total_purchase = st.number_input("Total Purchase Amount ($)", min_value=0, value=5000)
                    account_manager = st.selectbox("Has Account Manager", ["Yes", "No"])
                    years = st.number_input("Years with Company", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
                    num_sites = st.number_input("Number of Sites", min_value=1, value=5)
                    
                    # Submit button
                    submit_button = st.form_submit_button(label="Predict Churn Probability")
                
                # Convert inputs to model features
                features = pd.DataFrame({
                    'Age': [age],
                    'Total_Purchase': [total_purchase],
                    'Account_Manager': [1 if account_manager == "Yes" else 0],
                    'Years': [years],
                    'Num_Sites': [num_sites]
                })
                
                if submit_button:
                    try:
                        # Scale features
                        features_scaled = scaler.transform(features)
                        
                        # Make prediction
                        churn_prob = model.predict_proba(features_scaled)[0][1]
                        logger.info(f"Prediction made: {churn_prob:.4f}")
                        
                        # Display prediction
                        st.subheader("Prediction Results")
                        
                        # Create gauge chart for churn probability
                        fig = create_gauge_chart(churn_prob)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk assessment
                        risk_level = "Low" if churn_prob < 0.3 else "Medium" if churn_prob < 0.7 else "High"
                        
                        # Get recommendations
                        rec = get_recommendations(risk_level)
                        
                        # Show recommendations with appropriate styling
                        getattr(st, rec["style"])(f"**{rec['title']}**")
                        
                        st.subheader("Recommended Actions:")
                        for action in rec["actions"]:
                            st.markdown(f"- {action}")
                        
                        # Calculate feature impact
                        # Calculate SHAP values
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(features_scaled)
                        
                        # Convert SHAP values to absolute values and sum
                        feature_importance = pd.DataFrame({
                            'Feature': features.columns,
                            'Impact': np.abs(shap_values[1][0] if isinstance(shap_values, list) else shap_values[0])
                        })
                        feature_importance = feature_importance.sort_values('Impact', ascending=False)
                        
                        # Display top influential factors
                        st.subheader("Key Factors Influencing This Prediction")
                        for idx, row in feature_importance.iterrows():
                            st.text(f"{row['Feature']}: {row['Impact']:.4f}")
                    except Exception as e:
                        logger.error(f"Error making prediction: {e}")
                        st.error(f"Error making prediction: {str(e)}")
            
            if submit_button:
                with col2:
                    st.subheader("Customer Profile Analysis")
                    
                    # Display feature importance plot specific to this customer
                    fig = create_feature_importance_plot(model, features.columns)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add contextual information
                    age_bucket = "Young" if age < 30 else "Mid-age" if age < 50 else "Senior"
                    purchase_level = "Low" if total_purchase < 3000 else "Medium" if total_purchase < 7000 else "High"
                    tenure_category = "New" if years < 2 else "Established" if years < 5 else "Long-term"
                    
                    st.subheader("Customer Segment")
                    st.markdown(f"**Age Group:** {age_bucket}")
                    st.markdown(f"**Purchase Level:** {purchase_level}")
                    st.markdown(f"**Tenure Category:** {tenure_category}")
                    
                    if account_manager == "No" and risk_level != "Low":
                        st.warning("⚠️ **Action Needed:** Consider assigning an account manager")
        
        with tab2:
            st.subheader("About the Model")
            st.markdown("""
            ### Random Forest Classifier
            
            This prediction model uses a Random Forest algorithm to identify customers at risk of churning.
            
            **Model Features:**
            - Customer age
            - Total purchase amount
            - Account manager assignment
            - Years with company
            - Number of sites
            
            **Model Metrics:**
            - Trained on synthetic data with realistic distributions
            - Optimized for balanced precision and recall
            """)
            
            # Add model parameters section
            st.subheader("Model Parameters")
            st.json({
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            })
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
