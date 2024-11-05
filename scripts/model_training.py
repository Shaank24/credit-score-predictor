import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge  # Using Ridge Regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def split_data(X, y):
    # Split the data into training and testing sets
    print("Splitting data into training and testing sets.")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_ridge_regression(X_train, y_train, alpha=1.0):
    # Train a Ridge Regression model
    print("Training Ridge Regression model.")
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    print("Model training completed.")
    return ridge

def save_model(model, filename):
    # Save the trained model to disk
    joblib.dump(model, os.path.join('models', filename))
    print(f"Model saved as {filename}.")

if __name__ == "__main__":
    # Load preprocessed data
    X = pd.read_csv(os.path.join('data', 'X_preprocessed.csv'))
    y = pd.read_csv(os.path.join('data', 'y_preprocessed.csv'))
    y = y.values.ravel()  # Convert to 1D array

    print(f"Loaded preprocessed data. X shape: {X.shape}, y shape: {y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Train the model using Ridge Regression
    ridge_model = train_ridge_regression(X_train, y_train, alpha=1.0)

    # Evaluate the model
    y_pred = ridge_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Ridge Regression Model Performance on Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    # Load scaler_y for inverse transformation
    scaler_y = joblib.load(os.path.join('models', 'scaler_y.pkl'))
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    print("Sample Predictions (original scale):", y_pred_original[:5])
    print("Actual Values (original scale):", y_test_original[:5])

    # Print model coefficients
    print("Model coefficients:")
    coefficients = ridge_model.coef_
    for feature, coef in zip(X.columns, coefficients):
        print(f"{feature}: {coef}")

    # Save the model
    save_model(ridge_model, 'credit_score_model.pkl')

