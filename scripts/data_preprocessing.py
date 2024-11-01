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
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Fill numerical missing values with the mean
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Fill categorical missing values with the mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df

def encode_categorical_variables(df):
    """Encode categorical variables using OneHotEncoder."""
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude 'CUST_ID' and target variables
    cols_to_exclude = ['CUST_ID', 'CREDIT_SCORE', 'DEFAULT']  # Add any other columns to exclude
    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]

    print("Categorical columns to be encoded:", categorical_cols)

    if not categorical_cols:
        print("No categorical columns to encode.")
        return df, None  # Return df as is and None for encoder

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
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

    # Exclude 'CUST_ID' from features
    cols_to_drop = ['CUST_ID', 'CREDIT_SCORE', 'DEFAULT']
    X = df_encoded.drop(cols_to_drop, axis=1)
    y = df_encoded['CREDIT_SCORE']

    # Save expected features for future use
    expected_features = X.columns.tolist()
    joblib.dump(expected_features, 'models/expected_features.pkl')

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Save the encoder and scaler if they exist
    if encoder is not None:
        joblib.dump(encoder, 'models/encoder.pkl')
    else:
        print("No encoder to save.")

    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y

if __name__ == "__main__":
    # Define the file path
    file_path = os.path.join('data', 'credit_score.csv')

    # Preprocess the data
    X_scaled, y = preprocess_data(file_path)

    print("Shape of X_scaled:", X_scaled.shape)
    print("Shape of y:", y.shape)

    # Save the preprocessed data
    X_scaled.to_csv(os.path.join('data', 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join('data', 'y_preprocessed.csv'), index=False)

