import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_model_and_scalers():
    # Load the trained model and preprocessing objects
    model = joblib.load(os.path.join('models', 'credit_score_model.pkl'))
    scaler_X = joblib.load(os.path.join('models', 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join('models', 'scaler_y.pkl'))
    expected_features = joblib.load(os.path.join('models', 'expected_features.pkl'))
    print("Model and preprocessing objects loaded.")
    return model, scaler_X, scaler_y, expected_features

def preprocess_user_input(user_input, scaler_X, expected_features):
    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])
    print("User input DataFrame:")
    print(input_df)

    # Reorder columns to match the training data
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    print("Input features after reordering:")
    print(input_df.head())

    # Scale numerical features
    input_scaled_array = scaler_X.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled_array, columns=expected_features)

    print("Input features after scaling:")
    print(input_scaled.head())

    return input_scaled

def main():
    st.title("Credit Score Predictor")

    # Load model and preprocessing objects
    model, scaler_X, scaler_y, expected_features = load_model_and_scalers()

    # Collect user input
    st.header("Input Customer Information")

    user_input = {}
    for feature in expected_features:
        if feature == 'INCOME':
            user_input[feature] = st.number_input('Annual Income', min_value=0.0, value=50000.0)
        elif feature == 'DEBT':
            user_input[feature] = st.number_input('Total Debt', min_value=0.0, value=10000.0)
        elif 'R_' in feature:
            user_input[feature] = st.number_input(f'Enter {feature}', min_value=0.0, max_value=1.0, value=0.2)
        else:
            user_input[feature] = st.number_input(f'Enter {feature}', value=0.0)

    if st.button("Predict Credit Score"):
        try:
            # Preprocess user input
            input_scaled = preprocess_user_input(user_input, scaler_X, expected_features)

            # Make prediction
            prediction_scaled = model.predict(input_scaled)
            print(f"Scaled prediction: {prediction_scaled}")

            # Inverse transform the prediction to original scale
            prediction_original = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            print(f"Original scale prediction: {prediction_original}")

            # Display the prediction
            st.success(f"Predicted Credit Score: {prediction_original[0][0]:.2f}")

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.error(str(e))
            print("Error during prediction:")
            print(str(e))

if __name__ == "__main__":
    main()

