# Customer Churn Prediction Dashboard

A Streamlit-based machine learning application for predicting customer churn probability in subscription-based businesses.

## Features

- **Secure API Authentication**: Requires API key for access to prediction features
- **Real-time Churn Prediction**: Enter customer information and get instant churn probability predictions
- **Risk Assessment**: Automatically categorizes customers into Low, Medium, or High risk segments
- **Actionable Recommendations**: Provides tailored action plans for each risk category
- **Data Visualizations**: Uses interactive Plotly visualizations to explain predictions
- **Model Insights**: Displays feature importance and SHAP values to understand key factors affecting churn

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   streamlit run Churn.py
   ```
5. Enter your API key when prompted (for testing, enter any string longer than 10 characters)

## Project Structure

- `Churn.py`: Main application file with Streamlit UI and model logic
- `requirements.txt`: Required Python packages
- Model files (generated on first run):
  - `model.joblib`: Trained RandomForest model
  - `scaler.joblib`: StandardScaler for feature normalization
  - `encoder.joblib`: LabelEncoder for categorical variables

## Model Details

The application uses a Random Forest Classifier with the following features:
- Customer age
- Total purchase amount
- Account manager assignment status
- Years with company
- Number of sites

The model is initially trained on synthetic data if no model files are found, and can be replaced with a custom-trained model for production use.

## License

MIT License