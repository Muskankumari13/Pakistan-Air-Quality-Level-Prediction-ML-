import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Load model, encoders & scaler
# ------------------------------
model = joblib.load("aqi_model.pkl")
le_city = joblib.load("city_encoder.pkl")
le_target = joblib.load("aqi_encoder.pkl")

# Load scaler if used during training
try:
    scaler = joblib.load("feature_scaler.pkl")
    use_scaler = True
except:
    use_scaler = False  # fallback if no scaler was used

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("Pakistan Air Quality Predictor")

# Select city
city = st.selectbox("Select City", le_city.classes_)

# Input features
st.subheader("Input Features (Use realistic values)")
pm2_5_lag1 = st.number_input("Previous Day PM2.5", 0.0, value=150.0)
pm2_5_lag3 = st.number_input("PM2.5 (3 Days Ago)", 0.0, value=100.0)
pm2_5_lag7 = st.number_input("PM2.5 (7 Days Ago)", 0.0, value=110.0)
pm2_5_roll3 = st.number_input("3-Day Rolling PM2.5", 0.0, value=120.0)
pm2_5_roll7 = st.number_input("7-Day Rolling PM2.5", 0.0, value=110.0)
temperature = st.number_input("Temperature (°C)", value=36.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)
pm10 = st.number_input("PM10", 0.0, value=90.0)
no2 = st.number_input("NO2", 0.0, value=120.0)
o3 = st.number_input("O3", 0.0, value=50.0)
so2 = st.number_input("SO2", 0.0, value=25.0)
co = st.number_input("CO", 0.0, value=1.0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict AQI Category"):

    city_encoded = le_city.transform([city])[0]

    pm2_5_lags = [pm2_5_lag1, pm2_5_lag3, pm2_5_lag7]
    pm2_5_rolls = [pm2_5_roll3, pm2_5_roll7]

    category_to_pm = {
        "Good": 40,
        "Moderate": 80,
        "Unhealthy for Sensitive": 130,
        "Unhealthy": 180,
        "Very Unhealthy": 250
    }

    forecast_results = []

    for day in [1, 2, 3]:
        X_new = pd.DataFrame([[pm2_5_lags[0], pm2_5_lags[1], pm2_5_lags[2],
                               pm2_5_rolls[0], pm2_5_rolls[1],
                               temperature, humidity, wind_speed,
                               pm10, no2, o3, so2, co,
                               city_encoded]],
                             columns=[
                                 'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag7',
                                 'pm2_5_roll3', 'pm2_5_roll7',
                                 'temperature', 'humidity', 'wind_speed',
                                 'pm10', 'no2', 'o3', 'so2', 'co',
                                 'city_encoded'
                             ])

        # ------------------------------
        # Apply scaling if available
        # ------------------------------
        if use_scaler:
            X_input = scaler.transform(X_new)
        else:
            X_input = X_new.values

        # Prediction
        pred_enc = model.predict(X_input)[0]
        pred_cat = le_target.inverse_transform([pred_enc])[0]

        forecast_results.append({
            "Day": f"Day {day}",
            "Predicted AQI": pred_cat
        })

        # Update lag features for next day forecast
        pm_pred = category_to_pm.get(pred_cat, pm2_5_lags[0])
        pm2_5_lags = [pm_pred, pm2_5_lags[0], pm2_5_lags[1]]
        pm2_5_rolls = [np.mean(pm2_5_lags[:3]), np.mean(pm2_5_lags)]

    forecast_df = pd.DataFrame(forecast_results)

    # ------------------------------
    # Display Forecast
    # ------------------------------
    st.subheader(f"{city} – 3 Day AQI Forecast")
    st.table(forecast_df)

    # ------------------------------
    # Alerts + Health Advisory
    # ------------------------------
    st.subheader("🚨 Health Alerts")
    for _, row in forecast_df.iterrows():
        if row["Predicted AQI"] == "Good":
            st.success(f"{row['Day']}: Good – Safe for outdoor activities 🟢")
        elif row["Predicted AQI"] == "Moderate":
            st.warning(f"{row['Day']}: Moderate – Sensitive groups should be cautious 🟡")
        else:
            st.error(f"{row['Day']}: {row['Predicted AQI']} – Avoid outdoor exposure 🔴")

    # ------------------------------
    # AQI Trend Chart
    # ------------------------------
    st.subheader("📈 AQI Trend")
    category_map = {
        "Good": 1,
        "Moderate": 2,
        "Unhealthy for Sensitive": 3,
        "Unhealthy": 4,
        "Very Unhealthy": 5
    }
    forecast_df["AQI_Level"] = forecast_df["Predicted AQI"].map(category_map)
    st.line_chart(forecast_df.set_index("Day")["AQI_Level"])

    # ------------------------------
    # Feature Importance (if available)
    # ------------------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("🔍 Factors Affecting AQI")
        features = X_new.columns
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(7)

        st.bar_chart(fi_df.set_index("Feature"))

    # ------------------------------
    # City-wise Risk Ranking
    # ------------------------------
    st.subheader("🏙️ City-wise AQI Risk Ranking")
    all_cities = le_city.classes_
    risk_scores = []

    for c in all_cities:
        c_encoded = le_city.transform([c])[0]
        X_temp = pd.DataFrame([[pm2_5_lag1, pm2_5_lag3, pm2_5_lag7,
                                pm2_5_roll3, pm2_5_roll7,
                                temperature, humidity, wind_speed,
                                pm10, no2, o3, so2, co,
                                c_encoded]],
                              columns=X_new.columns)
        # Apply scaler if exists
        if use_scaler:
            X_temp_input = scaler.transform(X_temp)
        else:
            X_temp_input = X_temp.values

        pred_enc = model.predict(X_temp_input)[0]
        pred_cat = le_target.inverse_transform([pred_enc])[0]
        risk_scores.append({
            "City": c,
            "AQI_Category": pred_cat,
            "Risk_Level": category_map[pred_cat]
        })

    risk_df = pd.DataFrame(risk_scores).sort_values(by="Risk_Level", ascending=False)
    st.table(risk_df.reset_index(drop=True))

    # ------------------------------
    # Download CSV
    # ------------------------------
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Forecast CSV",
        csv,
        f"{city}_AQI_Forecast.csv",
        "text/csv"
    )
