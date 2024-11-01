# scripts/model_evaluation.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_test_data():
    """Load the test features and target variable."""
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    return X_test, y_test

def load_model(model_name):
    """Load the trained model from the models directory."""
    model_path = f'../models/{model_name}'
    model = joblib.load(model_path)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

if __name__ == "__main__":
    # Load test data
    X_test, y_test = load_test_data()

    # Load the trained model
    model = load_model('credit_score_model.pkl')

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

