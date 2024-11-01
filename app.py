import streamlit as st
import traceback
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

def encode_categorical_features(X, encoder):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        return X, encoder

    encoded_cols = encoder.transform(X[categorical_cols])
    encoded_features_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_cols, columns=encoded_features_names)
    X = X.drop(categorical_cols, axis=1).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    X = pd.concat([X, encoded_df], axis=1)
    return X, encoder

def preprocess_user_input(user_input, encoder, scaler_X, expected_features):
    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    input_encoded, _ = encode_categorical_features(input_df, encoder)

    for col in expected_features:
    	if col not in input_encoded.columns:
    		input_encoded[col] = 0 # Assign default values for missing features

    # Remove extra columns not in expected_features
    input_encoded = input_encoded[expected_features]

    # **New code to inspect feature names**
    st.write("Scaler's expected feature names:")
    st.write(list(scaler_X.feature_names_in_))

    st.write("Input encoded feature names:")
    st.write(list(input_encoded.columns))

    input_encoded = input_encoded[scaler_X.feature_names_in_]

    # Scale features
    input_scaled_array = scaler_X.transform(input_encoded)

    input_scaled = pd.DataFrame(input_scaled_array, columns=scaler_X.feature_names_in_)

    input_scaled = input_scaled[model.feature_names_in_]

    return input_scaled, input_encoded

# Load model and preprocessing objects
model, encoder, scaler_X, expected_features = load_model_and_scalers()

model_feature_names = model.feature_names_in_

st.write("Model's expected feature names:")
st.write(list(model_feature_names))

# Streamlit app layout
st.title('Credit Score Predictor')

st.header('Enter Customer Details')

# Collect user input
income = st.number_input('Annual Income', min_value=0, value=50000)
savings = st.number_input('Savings', min_value=0, value=10000)
debt = st.number_input('Debt', min_value=0, value=5000)
cat_gambling = st.selectbox('Gambling Category', options=['No', 'Low', 'High'])
cat_credit_card = st.selectbox('Has Credit Card?', options=[0, 1])
cat_mortgage = st.selectbox('Has Mortgage?', options=[0, 1])
cat_savings_account = st.selectbox('Has Savings Account?', options=[0, 1])
cat_dependents = st.selectbox('Has Dependents?', options=[0, 1])

# Calculate derived features
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

r_savings_income = safe_divide(savings, income)
r_debt_income = safe_divide(debt, income)
r_debt_savings = safe_divide(debt, savings)

# Create a dictionary of user input
user_input = {
    'INCOME': income,
    'SAVINGS': savings,
    'DEBT': debt,
    'R_SAVINGS_INCOME': r_savings_income,
    'R_DEBT_INCOME': r_debt_income,
    'R_DEBT_SAVINGS': r_debt_savings,
    'CAT_GAMBLING': cat_gambling,
    'CAT_CREDIT_CARD': cat_credit_card,
    'CAT_MORTGAGE': cat_mortgage,
    'CAT_SAVINGS_ACCOUNT': cat_savings_account,
    'CAT_DEPENDENTS': cat_dependents,
    # Include other features as needed
}

# Assign default values or collect inputs for transaction features
# For demonstration, we'll use training data means
X_train_means = pd.read_csv('data/X_preprocessed.csv').mean()
transaction_features = [col for col in expected_features if col not in user_input]
for feature in transaction_features:
    user_input[feature] = X_train_means.get(feature, 0)

if st.button('Predict Credit Score'):
    try:
        # Preprocess the input
        input_scaled, input_encoded = preprocess_user_input(user_input, encoder, scaler_X, expected_features)
	
        # Display the input features and scaled values
        st.write("Input DataFrame feature names:")
        st.write(list(input_scaled.columns))

        # Display the model's expected feature names
        model_feature_names = model.feature_names_in_
        st.write("Model's expected feature names:")
        st.write(list(model_feature_names))

        st.write("Expected features:")
        st.write(expected_features) 


        # Make prediction
        prediction = model.predict(input_scaled)

        # Display the prediction
        st.success(f'Predicted Credit Score: {prediction[0]:.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error(traceback.format_exc())	

