import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def split_data(X, y):
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    # Train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def save_model(model, filename):
    # Save the trained model to disk
    joblib.dump(model, os.path.join('models', filename))

if __name__ == "__main__":
    # Load preprocessed data
    X = pd.read_csv(os.path.join('data', 'X_preprocessed.csv'))
    y = pd.read_csv(os.path.join('data', 'y_preprocessed.csv'))
    y = y.values.ravel()  # Convert to 1D array
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model
    lr_model = train_linear_regression(X_train, y_train)
    
    # Save the model
    save_model(lr_model, 'credit_score_model.pkl')
    
    # Save test data for evaluation
    X_test.to_csv(os.path.join('data', 'X_test.csv'), index=False)
    pd.DataFrame(y_test, columns=['CREDIT_SCORE']).to_csv(os.path.join('data', 'y_test.csv'), index=False)
    
    # Evaluate the model
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Model Performance on Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")

    print("Sample Predictions:", y_pred[:5])
    print("Actual Values:", y_test[:5])
