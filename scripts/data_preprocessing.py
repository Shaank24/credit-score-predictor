# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """Load the dataset from the specified file path."""
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    # Fill numerical missing values with the mean
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Fill categorical missing values with the mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df

def encode_categorical_variables(df):
    """Encode categorical variables using OneHotEncoder."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    df = df.drop(categorical_cols, axis=1)
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    return df_encoded, encoder

def scale_features(X):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler

def preprocess_data(file_path):
    """Complete data preprocessing pipeline."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df_encoded, encoder = encode_categorical_variables(df)

    # Separate features and target variable
    X = df_encoded.drop(['CREDIT_SCORE', 'DEFAULT'], axis=1)
    y = df_encoded['CREDIT_SCORE']

    # Save expected features for future use
    expected_features = X.columns.tolist()
    joblib.dump(expected_features, 'models/expected_features.pkl')

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Save the encoder and scaler
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y

if __name__ == "__main__":
    # Define the file path
    file_path = os.path.join('..', 'data', 'credit_score.csv')

    # Preprocess the data
    X_scaled, y = preprocess_data(file_path)

    # Save the preprocessed data
    X_scaled.to_csv(os.path.join('data', 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join('data', 'y_preprocessed.csv'), index=False)

