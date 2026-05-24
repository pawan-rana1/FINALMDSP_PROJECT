# ============================================================
# app.py
# STREAMLIT BATTERY SOC PREDICTION WEB APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Battery SOC Predictor",
    page_icon="🔋",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/best_soc_model.pkl"

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ============================================================
# INITIALIZE
# ============================================================

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔋 Battery SOC Model")

st.sidebar.markdown("### Model Information")

st.sidebar.write("Model Type: XGBoost Regressor")
st.sidebar.write("Target: Battery SOC Prediction")
st.sidebar.write("Input Features:")
st.sidebar.write("- Voltage")
st.sidebar.write("- Current")
st.sidebar.write("- Temperature")
st.sidebar.write("- dV/dt")
st.sidebar.write("- Rolling Voltage")
st.sidebar.write("- Cumulative Current")
st.sidebar.write("- Charge State")

st.sidebar.markdown("---")

st.sidebar.write("### Performance")

st.sidebar.write("R² Score: 0.9912")

st.sidebar.markdown("---")

st.sidebar.write("Developed for Battery ML Research")

# ============================================================
# MAIN TITLE
# ============================================================

st.title("🔋 Battery State of Charge (SOC) Prediction")

st.markdown("""
This web application predicts the real-time
State of Charge (SOC) of a lithium-ion battery
using a trained Machine Learning model.
""")

# ============================================================
# CHECK MODEL
# ============================================================

if model is None:
    st.error("Model file not found.")
else:
    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.header("Battery Input Parameters")

    col1, col2 = st.columns(2)

    with col1:

        voltage = st.number_input(
            "Voltage (V)",
            min_value=0.0,
            max_value=5.0,
            value=3.70,
            step=0.01
        )

        current = st.number_input(
            "Current (A)",
            min_value=-100.0,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        dv_dt = st.number_input(
            "dV/dt (V/s)",
            value=0.0,
            step=0.0001,
            format="%.5f"
        )

    with col2:

        rolling_v10 = st.number_input(
            "Rolling Voltage (10-step)",
            value=3.70,
            step=0.01
        )

        rolling_v50 = st.number_input(
            "Rolling Voltage (50-step)",
            value=3.70,
            step=0.01
        )

        cumulative_ah = st.number_input(
            "Cumulative Current (Ah)",
            value=0.5,
            step=0.1
        )

        charge_state = st.selectbox(
            "Charge State",
            options=[1, 0, -1],
            format_func=lambda x:
            "Charging (1)" if x == 1
            else (
                "Resting (0)"
                if x == 0
                else "Discharging (-1)"
            )
        )

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button("Predict SOC"):

        # ====================================================
        # FEATURE ORDER
        # ====================================================

        feature_columns = [
            'Voltage',
            'Current',
            'Temperature',
            'dV_dt',
            'Rolling_Voltage_10',
            'Rolling_Voltage_50',
            'Cumulative_Current_Ah',
            'Charge_State'
        ]

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        features = pd.DataFrame([[
            voltage,
            current,
            temperature,
            dv_dt,
            rolling_v10,
            rolling_v50,
            cumulative_ah,
            charge_state
        ]], columns=feature_columns)

        # ====================================================
        # PREDICTION
        # ====================================================

        # XGBoost handles unscaled data perfectly, so we pass raw features
        prediction = model.predict(features)[0]

        prediction = max(0.0, min(100.0, prediction))

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(f"Predicted SOC: {prediction:.2f}%")

        st.metric(
            label="Battery SOC",
            value=f"{prediction:.2f}%"
        )

        st.progress(int(prediction))

        # ====================================================
        # BATTERY STATUS
        # ====================================================

        if prediction > 80:
            st.success("Battery Highly Charged")
        elif prediction > 40:
            st.warning("Battery Moderately Charged")
        else:
            st.error("Battery Low")

        # ====================================================
        # STORE HISTORY
        # ====================================================

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append(prediction)

        # ====================================================
        # HISTORY GRAPH
        # ====================================================

        history_df = pd.DataFrame(
            st.session_state.history,
            columns=["SOC"]
        )

        st.subheader("Prediction History")

        st.line_chart(history_df)

        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("Input Summary")
        st.dataframe(features)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
Developed for Research-Grade Battery SOC Estimation,
Battery Management Systems (BMS),
EV Battery Analytics,
and Digital Twin Research.
""")
