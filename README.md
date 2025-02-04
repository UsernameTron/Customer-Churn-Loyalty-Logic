# Customer Churn Prediction for Subscription Service
⚠️ **Security Notice:** This repository does not contain any sensitive or personal data. Please ensure that you do not commit any API keys, passwords, or confidential information.

## Overview
A Streamlit application that predicts customer churn for a subscription service using machine learning. Helps businesses identify customers at risk of leaving and take proactive measures to retain them.

## Features
- **Interactive Inputs:** Enter customer details such as Age, Total Purchase, Account Manager status, Years with the company, and Number of Sites.
- **Real-time Predictions:** Get instant churn predictions based on input data.
- **Model Performance Metrics:** View Accuracy, Precision, Recall, F1 Score, and ROC-AUC scores.
- **Feature Importance:** Understand which features influence churn the most.
- **SHAP Explanations:** Visualize individual prediction explanations.
- **Model Retraining:** Retrain the machine learning model directly from the app.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UsernameTron/Customer-Churn-Loyalty-Logic.git
   cd Customer-Churn-Loyalty-Logic

	2.	Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install Dependencies

pip install -r requirements.txt



Usage

Run the Streamlit app:

streamlit run Churn.py

The app will open in your default web browser at http://localhost:8501.

Files
	•	Churn.py: Main Streamlit application script.
	•	requirements.txt: List of Python dependencies.
	•	README.md: Project documentation.
	•	.gitignore: Specifies files and directories to be ignored by Git.
	•	customer_churn.csv: Dataset used for training and predictions.
	•	model.joblib, scaler.joblib, encoder.joblib: Serialized machine learning model and preprocessing objects.



