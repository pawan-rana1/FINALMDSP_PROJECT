# 🔋 Deep Learning Battery SOC Predictor (LSTM)

This repository contains an end-to-end Machine Learning and Deep Learning pipeline designed to predict the State of Charge (SOC) of a lithium-ion battery. The project bridges metallurgical engineering, thermal management, and computational data science by translating pure laboratory physics into a deployable Artificial Intelligence model.

## 🔬 The Physics & Feature Engineering
The raw data was generated using high-precision Arbin testing cyclers across a comprehensive thermal sweep (0°C, 25°C, and 45°C) to capture non-linear voltage sag and cold-weather electrolyte sluggishness. 

Because physical battery management systems (BMS) lack a direct "capacity sensor," the true SOC target was engineered mathematically using **Coulomb Counting (Ampere-Hour Integration)** coupled with **Min-Max Scaling** to map raw electron flow to a normalized 0.0 to 1.0 curve.

## 🛡️ Preventing Data Leakage
To ensure real-world viability, the engineered `Capacity(Ah)` variable was strictly omitted from the model inputs. The models were forced to learn the battery's degradation curves utilizing exclusively the hardware sensors available inside a real Electric Vehicle:
* **Voltage (V)**
* **Current (A)**
* **Temperature (°C)**
* **Time (s)**

## 🧠 The Deep Learning Model: LSTM
While Random Forest and XGBoost provided strong baseline predictions, SOC is inherently tied to historical momentum (current drawn over time). 

The final deployed architecture is a **Long Short-Term Memory (LSTM)** neural network. The 2D sensor data was reshaped into 3D time-windows, allowing the network to process the previous 10 seconds of continuous telemetry to formulate a highly accurate prediction of current capacity (achieving a **93.88% R² score**).

## 🚀 Live Streamlit Deployment (Rolling Buffer)
Because LSTMs require a sequence of data, the `app.py` script utilizes Streamlit's `session_state` to act as a **Rolling RAM Buffer**. It continuously holds the last 10 seconds of user-adjusted sensor input, perfectly mimicking how a real-world BMS queues data before execution.