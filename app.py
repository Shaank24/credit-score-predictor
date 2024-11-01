# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing objects
model = joblib.load('models/credit_score_model.pkl')
encoder = joblib.load('models/encoder.pkl')
scaler = joblib.load('models/scaler.pkl')
expected_features = joblib.load('models/expected_features.pkl')

# Function to preprocess user input
def preprocess_user_input(input_data):
	df = pd.DataFrame([input_data])

	# Encode categorical variables
	categorical_cols = ['CAT_GAMBLING']  # Replace with your actual categorical columns
	if categorical_cols:
		encoded_array = encoder.transform(df[categorical_cols])
		encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
		df = df.drop(categorical_cols, axis=1)
		df_encoded = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
	else:
		df_encoded = df

	# Ensure all expected features are present
	for col in expected_features:
        	if col not in df_encoded.columns:
            		df_encoded[col] = 0

    	# Reorder columns to match training data
	df_encoded = df_encoded[expected_features]

    	# Scale features
	df_scaled = scaler.transform(df_encoded)
	df_scaled = pd.DataFrame(df_scaled, columns=expected_features)

	return df_scaled

# Streamlit app layout
st.title('Credit Score Predictor')

st.header('Enter Customer Details')

# Collect user input
income = st.number_input('Annual Income', min_value=0, value=50000)
savings = st.number_input('Savings', min_value=0, value=10000)
debt = st.number_input('Debt', min_value=0, value=5000)
cat_gambling = st.selectbox('Gambling Category', options=['none', 'low', 'high'])
# Add input fields for all required features

# Create a dictionary of user input
user_input = {
    'INCOME': income,
    'SAVINGS': savings,
    'DEBT': debt,
    'CAT_GAMBLING': cat_gambling,
    # Add other features accordingly
}

if st.button('Predict Credit Score'):
    # Preprocess the input
    input_preprocessed = preprocess_user_input(user_input)

    # Make prediction
    prediction = model.predict(input_preprocessed)
    st.success(f'Predicted Credit Score: {prediction[0]:.2f}')

