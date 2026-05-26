import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(page_title="Live LSTM Battery SOC Predictor", layout="wide")
st.title("🔋 Real-Time LSTM Battery SOC Predictor")
st.markdown("Adjust the real-time hardware sensors below. The LSTM requires a **10-second history buffer** to predict momentum.")

# --- Load the AI Brain and Scaler ---
@st.cache_resource
def load_ai_components():
    # Load the trained LSTM model and its required scaler
    model = load_model('battery_soc_lstm_clean.keras')
    scaler = joblib.load('lstm_scaler_clean.pkl')
    return model, scaler

lstm_model, scaler = load_ai_components()

# --- Initialize the Rolling Buffer (Session State) ---
# We fill the first 10 seconds with "resting" baseline data so the LSTM has history
if 'history_buffer' not in st.session_state:
    # Structure: [Voltage(V), Current(A), Temperature(C), Time(s)]
    resting_data = [[3.8, 0.0, 25.0, i] for i in range(10)]
    st.session_state.history_buffer = resting_data
if 'current_time' not in st.session_state:
    st.session_state.current_time = 10

# --- User Interface: Real-Time Sensor Inputs ---
st.sidebar.header("🎛️ Real-Time Sensor Inputs")
new_voltage = st.sidebar.slider("Voltage (V)", min_value=2.5, max_value=4.2, value=3.8, step=0.01)
new_current = st.sidebar.slider("Current (A) [Negative = Drain]", min_value=-10.0, max_value=5.0, value=-2.0, step=0.1)
new_temp = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)

# --- The "Step Forward" Engine ---
if st.sidebar.button("⏱️ Submit Data (Step 1 Second)"):
    # 1. Advance the clock
    st.session_state.current_time += 1
    
    # 2. Create the new row of data from the sliders
    new_row = [new_voltage, new_current, new_temp, st.session_state.current_time]
    
    # 3. Add to the bottom of the buffer, and remove the oldest reading from the top
    st.session_state.history_buffer.append(new_row)
    st.session_state.history_buffer.pop(0)

# --- Display the Rolling Buffer ---
st.subheader("🧠 LSTM 10-Second Memory Buffer")
buffer_df = pd.DataFrame(
    st.session_state.history_buffer, 
    columns=["Voltage (V)", "Current (A)", "Temperature (°C)", "Time (s)"]
)
st.dataframe(buffer_df, use_container_width=True)

# --- Execute LSTM Prediction ---
st.markdown("---")
st.subheader("⚡ Live SOC Prediction")

# 1. Convert the buffer to a numpy array
raw_history = np.array(st.session_state.history_buffer)

# 2. Scale the data using the exact rules from training
scaled_history = scaler.transform(raw_history)

# 3. Reshape into the 3D Time-Window the LSTM demands: (1 sample, 10 timesteps, 4 features)
lstm_input = scaled_history.reshape(1, 10, 4)

# 4. Predict the State of Charge
prediction = lstm_model.predict(lstm_input)
current_soc = prediction[0][0] * 100  # Convert 0.0-1.0 to percentage

# 5. Display the Metric
st.metric(label="Current State of Charge (SOC)", value=f"{current_soc:.2f} %")