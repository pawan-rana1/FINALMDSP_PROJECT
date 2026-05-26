import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

# Page configurations
st.set_page_config(page_title="LSTM Battery BMS Dashboard", page_icon="🔋", layout="wide")

st.title("🔋 Real-Time Time-Series Battery SOC Tracker")
st.markdown("""
This production dashboard deploys a **Deep Long Short-Term Memory (LSTM) Recurrent Neural Network** scoring an R² of **93.88%**. 
Unlike standard snapshot calculators, this model evaluates the last **10 seconds of streaming telemetry history** to factor in chemical polarization, voltage sag, and thermal inertia.
""")

# Path checks for model assets
MODEL_PATH = "battery_soc_lstm_clean.keras"
SCALER_PATH = "lstm_scaler_clean.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(f"❌ Critical Deployment Error: Missing model files. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist in your repository folder.")
else:
    # Load LSTM assets
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    st.sidebar.header("📥 Telemetry Uplink Station")
    st.sidebar.write("LSTMs evaluate time-series sequences. Upload a consecutive log or generate synthetic hardware data below.")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Consecutive BMS Log (.csv)", type=["csv"])
    generate_sample = st.sidebar.button("⚙️ Generate Live Simulated Driving Log")
    
    df_raw = None
    
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("Log file loaded successfully!")
    elif generate_sample:
        # Generate 500 seconds of a realistic battery discharge trajectory
        sim_time = np.arange(0, 500, 1)
        sim_voltage = 4.15 - (0.75 * (sim_time / 500)**0.5) + np.random.normal(0, 0.01, len(sim_time))
        sim_current = -2.5 + np.random.normal(0, 0.5, len(sim_time)) 
        sim_temp = 25.0 + (5.0 * (sim_time / 500)) + np.random.normal(0, 0.1, len(sim_time)) 
        
        df_raw = pd.DataFrame({
            'Time(s)': sim_time,
            'Voltage(V)': sim_voltage,
            'Current(A)' : sim_current,
            'Temperature(C)': sim_temp
        })
        st.sidebar.info("Simulated hardware driving log synthesized.")

    if df_raw is not None:
        # Validate data schema
        required_cols = ['Voltage(V)', 'Current(A)', 'Temperature(C)', 'Time(s)']
        if not all(col in df_raw.columns for col in required_cols):
            st.error(f"❌ Schema Error! Uploaded file must contain these exact columns: {required_cols}")
        else:
            st.subheader("📊 Live Sensor Streams (Raw Incoming Logs)")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Extract features and scale them
            features = ['Voltage(V)', 'Current(A)', 'Temperature(C)', 'Time(s)']
            scaled_values = scaler.transform(df_raw[features])
            
            # Construct 3D windows (Lookback = 10 steps)
            time_steps = 10
            Xs = []
            for i in range(len(scaled_values) - time_steps):
                Xs.append(scaled_values[i:(i + time_steps)])
            X_3D = np.array(Xs)
            
            if len(X_3D) == 0:
                st.warning("⚠️ Log context is too short! Need at least 11 consecutive records.")
            else:
                with st.spinner("LSTM Engine running sequence inference..."):
                    preds = lstm_model.predict(X_3D).flatten()
                    preds_clamped = np.clip(preds * 100, 0.0, 100.0)
                
                tracking_timeline = df_raw['Time(s)'].values[time_steps:]
                live_voltage = df_raw['Voltage(V)'].values[time_steps:]
                
                st.markdown("---")
                st.subheader("🎯 Terminal Mission State Summary")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Final Timestamp", f"{tracking_timeline[-1]:.1f} s")
                m_col2.metric("Final Voltage", f"{live_voltage[-1]:.3f} V")
                m_col3.metric("LSTM Estimated SOC", f"{preds_clamped[-1]:.2f}%")
                
                st.markdown("---")
                st.subheader("📈 Integrated State Tracking System")
                
                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                color = '#1f77b4'
                ax1.set_xlabel('Time Processed (seconds)', fontweight='bold')
                ax1.set_ylabel('LSTM Predicted SOC (%)', color=color, fontweight='bold')
                ax1.plot(tracking_timeline, preds_clamped, color=color, linewidth=2.5, label='LSTM Tracking')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                ax2 = ax1.twinx()
                color = '#ff7f0e'
                ax2.set_ylabel('Hardware Measured Voltage (V)', color=color, fontweight='bold')
                ax2.plot(tracking_timeline, live_voltage, color=color, linestyle=':', linewidth=1.5, label='Voltage')
                ax2.tick_params(axis='y', labelcolor=color)
                
                fig.tight_layout()
                st.pyplot(fig)
    else:
        st.info("💡 Uplink Console Standby: Upload a driving telemetry file (.csv) or generate a log in the sidebar to test.")