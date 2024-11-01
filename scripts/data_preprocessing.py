# scripts/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from the specified file path."""
    df = pd.read_csv(filepath)
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
    """Encode categorical variables using one-hot encoding."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def scale_features(X):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled

def preprocess_data(filepath):
    """Complete data preprocessing pipeline."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df_encoded = encode_categorical_variables(df)
    X = df_encoded.drop(['CREDIT_SCORE', 'DEFAULT'], axis=1)
    y = df_encoded['CREDIT_SCORE']
    X_scaled = scale_features(X)
    return X_scaled, y

if __name__ == "__main__":
    # File path to the dataset
    filepath = 'data/credit_score.csv'

    # Preprocess the data
    X_scaled, y = preprocess_data(filepath)

    # Save the preprocessed data
    X_scaled.to_csv('data/X_preprocessed.csv', index=False)
    y.to_csv('data/y_preprocessed.csv', index=False)

