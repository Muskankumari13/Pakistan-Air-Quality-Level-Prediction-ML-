# Pakistan Air Quality Predictor

A **Streamlit web application** that predicts the **Air Quality Index (AQI)** for cities in Pakistan. The app forecasts AQI for the next **3 days**, provides **health alerts**, displays **AQI trends**, shows **important factors affecting air quality**, and ranks all cities based on **AQI risk**.

---

## Features

1. **City Selection**
   - Choose from major cities in Pakistan: Islamabad, Karachi, Lahore, Peshawar, Quetta, etc.

2. **Input Features**
   - Previous Day PM2.5
   - PM2.5 (3 Days Ago & 7 Days Ago)
   - 3-Day & 7-Day Rolling PM2.5
   - Temperature (°C)
   - Humidity (%)
   - Wind Speed (m/s)
   - PM10, NO2, O3, SO2, CO levels

3. **3-Day AQI Forecast**
   - Predicts AQI category for the next 3 days.
   - Categories: Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy.

4. **Health Alerts**
   - Provides safety guidance based on AQI:
     - 🟢 Good – Safe for outdoor activities
     - 🟡 Moderate – Sensitive groups should be cautious
     - 🔴 Unhealthy – Avoid outdoor exposure

5. **AQI Trend Visualization**
   - Line chart showing AQI levels over 3 days.

6. **Feature Importance**
   - Displays top features affecting AQI predictions (if model supports `feature_importances_`).

7. **City-wise Risk Ranking**
   - Ranks all cities based on predicted AQI risk level.
   - Helps identify the most polluted cities quickly.

8. **Download Forecast**
   - Download the 3-day AQI forecast as a CSV file.

---

## Installation

# 1. Clone the repository
https://github.com/Muskankumari13/Pakistan-Air-Quality-Level-Prediction-ML-
cd Pakistan-Air-Quality-Level-Prediction-ML

