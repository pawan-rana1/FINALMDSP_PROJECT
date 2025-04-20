import streamlit as st
import pandas as pd
import joblib

# Load model and mapping
model = joblib.load('model.pkl')
element_mapping = joblib.load('element_mapping.pkl')

# Invert mapping
reverse_mapping = {v: k for k, v in element_mapping.items()}

# UI
st.title("Lithium Battery Material Predictor")

# Inputs
element = st.selectbox("Element", list(element_mapping.keys()))
is_transition = st.selectbox("Is it a transition element?", ["Yes", "No"])
density = st.number_input("Density (must be positive)", min_value=0.0001, step=0.01)

if st.button("Predict"):
    encoded_element = element_mapping[element]
    is_transition_binary = 1 if is_transition == "Yes" else 0

    # Make prediction
    input_df = pd.DataFrame([[encoded_element, is_transition_binary, density]],
                            columns=['element_encoded', 'is_transition_element', 'density'])
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Output: {prediction}")
