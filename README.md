# Battery SOC Prediction System

## Overview

This project develops a Machine Learning-based
Battery State of Charge (SOC) Prediction System
using lithium-ion battery experimental datasets.

The system predicts real-time battery SOC using:

- Voltage
- Current
- Temperature
- Engineered battery features

The project is designed for:

- Battery Management Systems (BMS)
- Electric Vehicles (EVs)
- Digital Twin Systems
- Energy Storage Monitoring
- Battery Research Applications

---

# Features

- Automated battery dataset preprocessing
- SOC generation from capacity data
- Multi-temperature dataset integration
- Machine Learning SOC prediction
- Real-time Streamlit deployment
- Interactive SOC visualization

---

# Machine Learning Models

The following regression models were trained:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Best Model:
- XGBoost

Performance:
- R² Score: 0.9912

---

# Dataset Features

Input Features:

- Voltage
- Current
- Temperature
- dV/dt
- Rolling Voltage (10-step)
- Rolling Voltage (50-step)
- Cumulative Current
- Charge State

Output:

- Battery SOC (%)

---

# Project Structure

SOC-Prediction-Model/

├── app.py

├── requirements.txt

├── README.md

├── models/

│   ├── best_soc_model.pkl

├── data/

│   └── processed_battery_data.csv

├── notebooks/

│   └── SOC_Model.ipynb

└── src/

    ├── data_processing.py

    └── model_training.py

---

# Installation

Clone repository:

```bash
git clone <repository-link>

pip install -r requirements.txt
streamlit run app.py
```
