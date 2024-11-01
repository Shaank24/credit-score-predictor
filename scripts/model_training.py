# scripts/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_preprocessed_data():
    """Load preprocessed features and target variable."""
    X = pd.read_csv('data/X_preprocessed.csv')
    y = pd.read_csv('data/y_preprocessed.csv')

    # Ensure y is a panda Series
    if isinstance(y, pd.DataFrame):
    	y = y.squeeze() # Converts DataFrame to Series if it has one column
    return X, y

def split_data(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ensure y_train and y_test are Series
    if isinstance(y_train, pd.DataFrame):
    	y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame):
    	y_test = y_test.squeeze()

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def save_model(model, model_name):
    """Save the trained model to the models directory."""
    model_path = f'models/{model_name}'
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # Load preprocessed data
    X, y = load_preprocessed_data()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    lr_model = train_linear_regression(X_train, y_train)

    # Save the model
    save_model(lr_model, 'credit_score_model.pkl')

    # Save test data for evaluation
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

