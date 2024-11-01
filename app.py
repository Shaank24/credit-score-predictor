import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_model_and_scalers():
    # Load the trained model and preprocessing objects
    model = joblib.load(os.path.join('models', 'credit_score_model.pkl'))
    encoder = joblib.load(os.path.join('models', 'encoder.pkl'))
    scaler_X = joblib.load(os.path.join('models', 'scaler_X.pkl'))
    expected_features = joblib.load(os.path.join('models', 'expected_features.pkl'))
    return model, encoder, scaler_X, expected_features

def preprocess_user_input(user_input, encoder, scaler_X, expected_features):
    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical features
    input_encoded, _ = encode_categorical_features(input_df, encoder)
    
    # Load training data mean
    X_train_mean = pd.read_csv('data/X_preprocessed.csv').mean()

    # Ensure the DataFrame has all expected features
    for col in expected_features:
        if col not in input_encoded.columns:
            input_encoded[col] = X_train_mean[col]  # Use mean
    
    # Reorder columns to match the training data
    input_encoded = input_encoded[expected_features]
    
    # Scale features
    input_scaled = scaler_X.transform(input_encoded)
    
    return input_scaled

def encode_categorical_features(X, encoder):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        return X, encoder
    
    encoded_cols = encoder.transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return X, encoder

# Load model and preprocessing objects
model, encoder, scaler_X, expected_features = load_model_and_scalers()

# Streamlit app layout
st.title('Credit Score Predictor')

st.header('Enter Customer Details')

# Collect user input
income = st.number_input('Annual Income', min_value=0, value=50000)
savings = st.number_input('Savings', min_value=0, value=10000)
debt = st.number_input('Debt', min_value=0, value=5000)
cat_gambling = st.selectbox('Gambling Category', options=['none', 'low', 'high'])
cat_credit_card = st.selectbox('Has Credit Card?', options=[0, 1])
cat_mortgage = st.selectbox('Has Mortgage?', options=[0, 1])
cat_savings_account = st.selectbox('Has Savings Account?', options=[0, 1])
cat_dependents = st.selectbox('Has Dependents?', options=[0, 1])

# Create a dictionary of user input
user_input = {
    'INCOME': income,
    'SAVINGS': savings,
    'DEBT': debt,
    'CAT_GAMBLING': cat_gambling,
    'CAT_CREDIT_CARD': cat_credit_card,
    'CAT_MORTGAGE': cat_mortgage,
    'CAT_SAVINGS_ACCOUNT': cat_savings_account,
    'CAT_DEPENDENTS': cat_dependents,
}

if st.button('Predict Credit Score'):
    # Preprocess the input
    input_scaled = preprocess_user_input(user_input, encoder, scaler_X, expected_features)

    input_scaled_df = pd.DataFrame(input_scaled, columns=expected_features)
    
    st.write("Input Features and Scaled Values:")
    st.write(input_scaled_df.T)    
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.success(f'Predicted Credit Score: {prediction[0]:.2f}')

