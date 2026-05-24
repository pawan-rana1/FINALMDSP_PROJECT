import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/best_soc_model.pkl"

st.set_page_config(page_title="Battery SOC Predictor", page_icon="🔋", layout="centered")

st.title("🔋 Battery State of Charge (SOC) Prediction")
st.markdown("This application predicts the real-time SOC of a lithium-ion battery using a machine learning model.")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("Model not found. Please run model training first.")
else:
    st.header("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        voltage = st.number_input("Voltage (V)", value=3.7, step=0.01)
        current = st.number_input("Current (A)", value=1.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", value=25.0, step=1.0)
        dv_dt = st.number_input("dV/dt (V/s)", value=0.0, step=0.0001, format="%.5f")
        
    with col2:
        rolling_v10 = st.number_input("Rolling Voltage (10 steps)", value=3.7, step=0.01)
        rolling_v50 = st.number_input("Rolling Voltage (50 steps)", value=3.7, step=0.01)
        cumulative_ah = st.number_input("Cumulative Current (Ah)", value=0.5, step=0.1)
        charge_state = st.selectbox("Charge State", options=[1, 0, -1], format_func=lambda x: "Charging (1)" if x == 1 else ("Resting (0)" if x == 0 else "Discharging (-1)"))

    if st.button("Predict SOC"):
        features = pd.DataFrame([{
            'Voltage': voltage,
            'Current': current,
            'Temperature': temperature,
            'dV_dt': dv_dt,
            'Rolling_Voltage_10': rolling_v10,
            'Rolling_Voltage_50': rolling_v50,
            'Cumulative_Current_Ah': cumulative_ah,
            'Charge_State': charge_state
        }])
        
        prediction = model.predict(features)[0]
        
        # Clip between 0 and 100
        prediction = max(0.0, min(100.0, prediction))
        
        st.success(f"### Predicted SOC: {prediction:.2f}%")
        st.progress(int(prediction))

st.markdown("---")
st.markdown("Developed for Research-Grade Battery SOC Estimation.")
