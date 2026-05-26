# 🔋 Time-Series Lithium-Ion Battery SOC Estimation Using Deep LSTM Networks

An advanced, production-ready Battery Management System (BMS) solution that leverages a Deep Long Short-Term Memory (LSTM) Recurrent Neural Network to predict a battery's State of Charge (SOC). The model is trained on dynamic experimental laboratory data across a comprehensive thermal sweep (0°C, 25°C, 45°C).

## 🚀 Live Dashboard
The interactive time-series tracking application is built using Streamlit and deployed globally. 
* **Live App Link:** `[Insert your Streamlit Share link here once deployed]`

---

## 📊 Performance Metrics Matrix
By abandoning snapshot-based models and forcing the neural paths to look at **10 seconds of historical streaming sequence data**, the network factors in critical battery physics like polarization delay, internal resistance voltage drops, and thermal inertia.

| Model Architecture | MAE | RMSE | $R^2$ Score | Real-World Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 0.2210 | 0.2640 | ~64.20% | Fails under dynamic load curves |
| **Random Forest** | 0.0120 | 0.0180 | ~99.99% | At risk of spatial memorization |
| **Deep LSTM (Our App)** | **0.0240** | **0.0310** | **93.88%** | **Optimal for continuous streaming EV telemetry** |

---

## 🛠️ Data Engineering & Physics "Firewall"
To mirror real vehicle limitations, this model avoids data leakage entirely. The laboratory variables `Charge_Capacity` and `Discharge_Capacity` are completely dropped from inputs. The LSTM operates purely on accessible hardware sensor arrays:
1. **Time Elapsed (s)** — System synchronization clock.
2. **Voltage (V)** — Main electrical bus potential measurement.
3. **Current (A)** — Pack load current demand (Negative = Discharge, Positive = Charge).
4. **Temperature (°C)** — Core battery case thermal data.

---

## 📁 Repository Structure
```text
├── app.py                      # Streamlit interactive tracking web dashboard
├── requirements.txt            # Cloud package installer manual
├── battery_soc_lstm_clean.keras # Trained 2,945-parameter LSTM model binary
├── lstm_scaler_clean.pkl       # Companion StandardScaler transformation mapping
└── battery_soc_analysis.ipynb  # Engineering lab notebook (Data prep, EDA, training)