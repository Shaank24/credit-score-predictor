import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def encode_categorical_features(X, encoder=None):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        return X, encoder
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(X[categorical_cols])
    else:
        encoded_cols = encoder.transform(X[categorical_cols])
    
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return X, encoder

def scale_features(X, scaler=None):
    from sklearn.preprocessing import StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler

def preprocess_data(file_path):
    # Load data
    data = load_data(file_path)
    
    # Separate features and target
    X = data.drop(['CUST_ID', 'CREDIT_SCORE', 'DEFAULT'], axis=1)
    y = data['CREDIT_SCORE']
    
    # Encode categorical features
    X_encoded, encoder = encode_categorical_features(X)
    
    # Save the encoder
    if encoder is not None:
        joblib.dump(encoder, 'models/encoder.pkl')
    else:
        print("No encoder to save.")
    
    # Save expected feature order
    expected_features = X_encoded.columns.tolist()
    joblib.dump(expected_features, 'models/expected_features.pkl')
    
    # Scale features
    X_scaled, scaler_X = scale_features(X_encoded)
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    
    # Scale target variable
    y = y.values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    return X_scaled, y_scaled

if __name__ == "__main__":
    # Define the file path
    file_path = os.path.join('data', 'credit_score.csv')
    
    # Preprocess the data
    X_scaled, y_scaled = preprocess_data(file_path)
    
    print("Shape of X_scaled:", X_scaled.shape)
    print("Shape of y_scaled:", y_scaled.shape)
    
    # Save the preprocessed data
    X_scaled.to_csv(os.path.join('data', 'X_preprocessed.csv'), index=False)
    pd.DataFrame(y_scaled, columns=['CREDIT_SCORE']).to_csv(os.path.join('data', 'y_preprocessed.csv'), index=False)

