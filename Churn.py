# Churn.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from io import StringIO

# --------------------------
# Configuration and Setup
# --------------------------

# Set page configuration for responsive design
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Paths for model and scaler
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
ENCODER_PATH = "encoder.joblib"

# --------------------------
# Helper Functions
# --------------------------

@st.cache_data
def load_data(filepath):
    """
    Load the customer churn dataset.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def preprocess_data(df):
    """
    Preprocess the data by selecting relevant features and encoding categorical variables.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        tuple: Preprocessed DataFrame and fitted OneHotEncoder.
    """
    try:
        # Select relevant features
        selected_features = ["Age", "Total_Purchase", "Account_Manager", "Years", "Num_Sites", "Churn"]
        df = df[selected_features]

        # Handle categorical variables
        # Assuming 'Account_Manager' is binary; if more categorical features exist, handle them here
        encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        df['Account_Manager'] = df['Account_Manager'].astype(int)
        encoder.fit(df[['Account_Manager']])
        encoded = encoder.transform(df[['Account_Manager']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Account_Manager']))
        df = pd.concat([df.drop('Account_Manager', axis=1), encoded_df], axis=1)

        logging.info("Data preprocessed successfully.")
        return df, encoder
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        st.error(f"Error in preprocessing data: {e}")
        return None, None


@st.cache_resource
def load_or_train_model(df, _encoder):
    """
    Load the trained model and scaler if they exist; otherwise, train a new model.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        _encoder (OneHotEncoder): Fitted OneHotEncoder for categorical variables.

    Returns:
        tuple: Trained model, scaler, encoder, and evaluation metrics.
    """
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoder = joblib.load(ENCODER_PATH)
            logging.info("Model, scaler, and encoder loaded from disk.")

            # Split data for evaluation
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Transform test data using loaded scaler
            X_test_scaled = scaler.transform(X_test)

            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            }

            return model, scaler, encoder, metrics

        else:
            # Split data
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Initialize scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define model and hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
            grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
            grid.fit(X_train_scaled, y_train)
            model = grid.best_estimator_

            # Save model and scaler
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            joblib.dump(_encoder, ENCODER_PATH)  # Save the passed _encoder
            logging.info("Model trained and saved to disk.")

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            }

            return model, scaler, _encoder, metrics

    except Exception as e:
        logging.error(f"Error in loading or training model: {e}")
        st.error(f"Error in loading or training model: {e}")
        return None, None, None, None


def user_input_features(df, encoder):
    """
    Collect user input features through the sidebar.

    Args:
        df (pd.DataFrame): Original DataFrame.
        encoder (OneHotEncoder): Trained encoder for categorical variables.

    Returns:
        pd.DataFrame: User input features as DataFrame.
    """
    try:
        st.sidebar.header("Enter Customer Information")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            age = st.slider(
                'Age',
                min_value=int(df['Age'].min()),
                max_value=int(df['Age'].max()),
                value=int(df['Age'].mean()),
                help="Select the customer's age."
            )
            total_purchase = st.slider(
                'Total Purchase ($)',
                min_value=float(df['Total_Purchase'].min()),
                max_value=float(df['Total_Purchase'].max()),
                value=float(df['Total_Purchase'].mean()),
                help="Total amount purchased by the customer."
            )

        with col2:
            account_manager = st.selectbox(
                'Account Manager',
                options=[0, 1],
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help="Does the customer have an account manager?"
            )
            years = st.slider(
                'Years with the company',
                min_value=float(df['Years'].min()),
                max_value=float(df['Years'].max()),
                value=float(df['Years'].mean()),
                help="Number of years the customer has been with the company."
            )
            num_sites = st.slider(
                'Number of Sites',
                min_value=int(df['Num_Sites'].min()),
                max_value=int(df['Num_Sites'].max()),
                value=int(df['Num_Sites'].mean()),
                help="Number of sites the customer is associated with."
            )

        # Validation
        if years < 0:
            st.sidebar.error("Years with the company cannot be negative.")
            return None

        # Create DataFrame
        data = {
            'Age': age,
            'Total_Purchase': total_purchase,
            'Account_Manager': account_manager,
            'Years': years,
            'Num_Sites': num_sites
        }
        features = pd.DataFrame(data, index=[0])

        # Encode categorical variable
        encoded_account_manager = encoder.transform(features[['Account_Manager']])
        encoded_df = pd.DataFrame(
            encoded_account_manager,
            columns=encoder.get_feature_names_out(['Account_Manager'])
        )
        features = pd.concat([features.drop('Account_Manager', axis=1), encoded_df], axis=1)

        logging.info("User input collected successfully.")
        return features

    except Exception as e:
        logging.error(f"Error in user input features: {e}")
        st.error(f"An error occurred while collecting input features: {e}")
        return None


def display_metrics(metrics):
    """
    Display model evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    st.subheader("Model Performance Metrics")
    cols = st.columns(len(metrics))
    for idx, (key, value) in enumerate(metrics.items()):
        cols[idx].metric(key, f"{value:.2f}")


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on model coefficients.

    Args:
        model: Trained model.
        feature_names (list): List of feature names.
    """
    try:
        coef = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef
        }).sort_values(by='Coefficient', ascending=False)

        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance based on Coefficients')
        st.pyplot(fig)
    except Exception as e:
        logging.error(f"Error in plotting feature importance: {e}")
        st.error(f"An error occurred while plotting feature importance: {e}")


def shap_explanation(model, scaler, input_data, background_data):
    """
    Provide SHAP explanation for the prediction.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        input_data (pd.DataFrame): User input data.
        background_data (pd.DataFrame): Background data for SHAP explainer.
    """
    try:
        # Initialize SHAP explainer with background data
        explainer = shap.Explainer(model, scaler.transform(background_data))
        shap_values = explainer(scaler.transform(input_data))

        st.subheader("Prediction Explanation (SHAP)")

        # Create a new matplotlib figure
        fig = plt.figure(figsize=(10, 6))

        # Generate SHAP waterfall plot without 'ax' parameter
        shap.plots.waterfall(shap_values[0], show=False)

        # Render the plot with Streamlit
        st.pyplot(fig)

    except Exception as e:
        logging.error(f"Error in SHAP explanation: {e}")
        st.error(f"An error occurred while generating SHAP explanation: {e}")


def display_prediction(prediction, prediction_proba):
    """
    Display the prediction result and probabilities.

    Args:
        prediction (int): Prediction outcome.
        prediction_proba (array): Prediction probabilities.
    """
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.error("This customer is **likely to churn**.")
    else:
        st.success("This customer is **not likely to churn**.")

    st.subheader("Prediction Probability")
    st.write(f"**Probability of Not Churning:** {prediction_proba[0][0]:.2f}")
    st.write(f"**Probability of Churning:** {prediction_proba[0][1]:.2f}")


def plot_feature_distributions(df):
    """
    Plot distributions of features.

    Args:
        df (pd.DataFrame): DataFrame to plot.
    """
    st.subheader("Feature Distributions")
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    for feature in numeric_features[:-1]:  # Exclude target variable
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
        st.pyplot(fig)


def plot_correlation_matrix(df):
    """
    Plot the correlation matrix.

    Args:
        df (pd.DataFrame): DataFrame to plot.
    """
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


def feedback_form():
    """
    Provide a feedback form for users.
    """
    st.sidebar.header("Feedback")
    feedback = st.sidebar.text_area("Let us know your thoughts or any issues you encountered:")
    if st.sidebar.button("Submit Feedback"):
        if feedback:
            with open("feedback.txt", "a") as f:
                f.write(feedback + "\n")
            st.sidebar.success("Thank you for your feedback!")
            logging.info("User submitted feedback.")
        else:
            st.sidebar.error("Please enter your feedback before submitting.")


def documentation():
    """
    Provide documentation and user guide in the sidebar.
    """
    st.sidebar.header("Documentation")
    st.sidebar.markdown("""
    ### How to Use This App
    1. **Input Customer Information**: Use the sliders and dropdowns in the sidebar to enter the customer’s details.
    2. **View Predictions**: After entering the information, the app will display whether the customer is likely to churn.
    3. **Understand Model Performance**: Review the model's accuracy and other performance metrics to gauge its reliability.
    4. **Explore Feature Importance**: Check which features most influence the prediction.
    5. **Retrain the Model**: If you have new data, click the 'Retrain Model' button to update the model.
    """)


# --------------------------
# Main Application
# --------------------------

def main():
    st.title("🚀 Customer Churn Prediction for Subscription Service")

    # Load data
    data = load_data("customer_churn.csv")
    if data is None:
        st.stop()

    # Preprocess data
    df, encoder = preprocess_data(data)
    if df is None or encoder is None:
        st.stop()

    # Display data visualizations
    with st.expander("📊 Data Exploration"):
        plot_feature_distributions(df)
        plot_correlation_matrix(df)

    # Load or train model
    model, scaler, encoder, metrics = load_or_train_model(df, encoder)
    if model is None:
        st.stop()

    # Display model performance metrics
    display_metrics(metrics)

    # Feature importance
    feature_names = df.drop("Churn", axis=1).columns.tolist()
    plot_feature_importance(model, feature_names)

    # User input features
    input_df = user_input_features(data, encoder)
    if input_df is not None:
        st.subheader("🔍 Customer Input Information")
        st.write(input_df)

        # Make prediction
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display prediction
        display_prediction(prediction, prediction_proba)

        # SHAP explanation using a subset of the training data as background
        # **Important Fix:** Exclude 'Churn' from background data
        background = df.drop("Churn", axis=1).sample(n=100, random_state=42)  # Adjust n as needed
        shap_explanation(model, scaler, input_df, background)

    # Feedback form
    feedback_form()

    # Documentation
    documentation()

    # Option to retrain the model
    st.sidebar.header("Model Management")
    if st.sidebar.button('🔄 Retrain Model'):
        with st.spinner('Retraining the model...'):
            model, scaler, encoder, metrics = load_or_train_model(df, encoder)
            st.success('✅ Model retrained successfully!')
            display_metrics(metrics)
            plot_feature_importance(model, feature_names)
            logging.info("Model retrained upon user request.")


if __name__ == "__main__":
    main()