import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print("First 5 rows of the data:")
    print(data.head())
    return data

def encode_categorical_features(X, encoder=None):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns to encode: {categorical_cols}")

    if not categorical_cols:
        return X, encoder

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(X[categorical_cols])
    else:
        encoded_cols = encoder.transform(X[categorical_cols])

    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    print("Encoded categorical features:")
    print(encoded_df.head())

    X = X.drop(categorical_cols, axis=1).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    X_encoded = pd.concat([X, encoded_df], axis=1)
    print("First 5 rows of X after encoding:")
    print(X_encoded.head())
    return X_encoded, encoder

def scale_features(X, scaler=None):
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    one_hot_cols = [col for col in X.columns if col not in numerical_cols]

    if scaler is None:
        scaler = RobustScaler()
        X_scaled_array = scaler.fit_transform(X[numerical_cols])
    else:
        X_scaled_array = scaler.transform(X[numerical_cols])

    X_scaled = pd.DataFrame(X_scaled_array, columns=numerical_cols)
    X_scaled = pd.concat([X_scaled, X[one_hot_cols].reset_index(drop=True)], axis=1)

    print("First 5 rows of X after scaling:")
    print(X_scaled.head())
    return X_scaled, scaler

def scale_target(y, scaler=None):
    if scaler is None:
        scaler = RobustScaler()
        y_scaled_array = scaler.fit_transform(y.values.reshape(-1, 1))
    else:
        y_scaled_array = scaler.transform(y.values.reshape(-1, 1))
    y_scaled = pd.Series(y_scaled_array.flatten(), name=y.name)
    print("First 5 values of y after scaling:")
    print(y_scaled.head())
    return y_scaled, scaler

if __name__ == "__main__":
    # Load the data
    data = load_data(os.path.join('data', 'credit_score.csv'))

    # Define the target column
    target_column = 'CREDIT_SCORE'  # Ensure this matches your dataset

    # Exclude 'CUST_ID' and 'DEFAULT' from features
    data = data.drop(['CUST_ID', 'DEFAULT'], axis=1, errors='ignore')
    print("'CUST_ID' and 'DEFAULT' columns dropped.")

    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    print("Features and target separated.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Encode categorical features
    X_encoded, encoder = encode_categorical_features(X)

    # Feature selection using Random Forest importance
    print("Performing feature selection using Random Forest.")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_encoded, y)
    importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
    importances.sort_values(ascending=False, inplace=True)
    print("Feature importances from Random Forest:")
    print(importances)

    # Select top N features
    top_n = 7  # You can adjust this number
    top_features = importances.head(top_n).index.tolist()
    print(f"Selected top {top_n} features:")
    print(top_features)

    X_selected = X_encoded[top_features]

    # Save expected feature names for later use
    expected_features = X_selected.columns.tolist()
    joblib.dump(expected_features, os.path.join('models', 'expected_features.pkl'))
    print("Expected features saved.")

    # Scale features
    X_scaled, scaler_X = scale_features(X_selected)

    # Scale target variable
    y_scaled, scaler_y = scale_target(y)

    # Save the scalers and encoder
    joblib.dump(encoder, os.path.join('models', 'encoder.pkl'))
    joblib.dump(scaler_X, os.path.join('models', 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join('models', 'scaler_y.pkl'))
    print("Scalers and encoder saved.")

    # Save preprocessed data
    X_scaled.to_csv(os.path.join('data', 'X_preprocessed.csv'), index=False)
    y_scaled.to_csv(os.path.join('data', 'y_preprocessed.csv'), index=False)
    print("Preprocessed data saved.")

    print(f"Shape of X_scaled: {X_scaled.shape}")
    print(f"Shape of y_scaled: {y_scaled.shape}")

