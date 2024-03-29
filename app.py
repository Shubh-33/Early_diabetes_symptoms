import streamlit as st
import pickle
import pandas as pd

# Load the trained models
with open('logistic_regression_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to make predictions
def predict_diabetes(model, data):
    prediction = model.predict(data)
    return prediction

# Streamlit App
def main():
    st.title('Diabetes Prediction')

    # User input for features
    age = st.slider('Age', 0, 100, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    polyuria = st.radio('Polyuria', ['Yes', 'No'])
    polydipsia = st.radio('Polydipsia', ['Yes', 'No'])
    
    # Add other features similarly

    # Preprocess user input
    gender_encoded = 1 if gender == 'Male' else 0
    polyuria_encoded = 1 if polyuria == 'Yes' else 0
    polydipsia_encoded = 1 if polydipsia == 'Yes' else 0
    # Encode other features similarly

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Polyuria': [polyuria_encoded],
        'Polydipsia': [polydipsia_encoded]
        # Add other features similarly
    })

    # Display the input data
    st.subheader('Input Data')
    st.write(input_data)

    # Make predictions
    lr_prediction = predict_diabetes(lr_model, input_data)
    rf_prediction = predict_diabetes(rf_model, input_data)

    # Display predictions
    st.subheader('Prediction')
    st.write('Logistic Regression Prediction:', 'Positive' if lr_prediction[0] == 1 else 'Negative')
    st.write('Random Forest Prediction:', 'Positive' if rf_prediction[0] == 1 else 'Negative')

if __name__ == '__main__':
    main()
