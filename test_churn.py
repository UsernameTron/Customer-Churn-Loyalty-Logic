import pytest
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import functions to test
from Churn import (
    load_or_create_model,
    create_feature_importance_plot,
    get_recommendations,
    create_gauge_chart,
    validate_api_key
)

# Test model creation and loading
def test_load_or_create_model():
    # Remove existing model files if they exist
    for file in ['model.joblib', 'scaler.joblib', 'encoder.joblib']:
        if os.path.exists(file):
            os.remove(file)
    
    # Create new model
    model, scaler, encoder = load_or_create_model()
    
    # Check model type
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(scaler, StandardScaler)
    
    # Check if model files were created
    assert os.path.exists('model.joblib')
    assert os.path.exists('scaler.joblib')
    assert os.path.exists('encoder.joblib')
    
    # Load existing model
    model2, scaler2, encoder2 = load_or_create_model()
    
    # Ensure the same model was loaded
    assert model.get_params() == model2.get_params()

# Test feature importance plot creation
def test_create_feature_importance_plot():
    # Create test data
    feature_names = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']
    
    # Create a simple model
    model = RandomForestClassifier(random_state=42, n_estimators=5)
    X = np.random.random((10, 5))
    y = np.random.choice([0, 1], 10)
    model.fit(X, y)
    
    # Create plot
    fig = create_feature_importance_plot(model, feature_names)
    
    # Check plot type and content
    assert fig is not None
    assert len(fig.data) == 1
    
    # Check feature names
    plot_features = fig.data[0].y.tolist()
    assert set(plot_features) == set(feature_names)

# Test recommendation system
def test_get_recommendations():
    # Test high risk
    high_rec = get_recommendations("High")
    assert high_rec["style"] == "warning"
    assert len(high_rec["actions"]) >= 3
    
    # Test medium risk
    med_rec = get_recommendations("Medium")
    assert med_rec["style"] == "info"
    assert len(med_rec["actions"]) >= 3
    
    # Test low risk
    low_rec = get_recommendations("Low")
    assert low_rec["style"] == "success"
    assert len(low_rec["actions"]) >= 3
    
    # Test invalid risk
    invalid_rec = get_recommendations("Unknown")
    assert invalid_rec == {}

# Test gauge chart creation
def test_create_gauge_chart():
    # Test with different probability values
    for prob in [0.1, 0.5, 0.9]:
        fig = create_gauge_chart(prob)
        
        # Check figure properties
        assert fig is not None
        assert fig.data[0].value == prob * 100
        assert fig.data[0].gauge.axis.range == [0, 100]

# Test API key validation
def test_validate_api_key():
    # Test invalid keys
    assert validate_api_key("") == False
    assert validate_api_key("short") == False
    
    # Test valid key
    assert validate_api_key("valid-api-key-12345") == True
    assert validate_api_key("thisisavalidtestapikey") == True

if __name__ == "__main__":
    pytest.main(["-xvs", "test_churn.py"])