import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models and scaler
models = {
    "p10": joblib.load("app/models/xgboost_p10_model.pkl"),
    "p50": joblib.load("app/models/xgboost_p50_model.pkl"),
    "p90": joblib.load("app/models/xgboost_p90_model.pkl")
}
scaler = joblib.load("app/models/scaler.pkl")

# Function for user input form


def user_input():
    st.sidebar.header("Input Features")
    hour = st.sidebar.slider("Hour", 0, 23, 12)
    dayofweek = st.sidebar.slider("Day of Week", 0, 6, 2)
    month = st.sidebar.slider("Month", 1, 12, 6)
    price_lag_1 = st.sidebar.number_input("Price Lag 1", value=159.371647)
    price_lag_2 = st.sidebar.number_input("Price Lag 2", value=159.371647)
    price_lag_7 = st.sidebar.number_input("Price Lag 7", value=159.371647)
    price_ma7 = st.sidebar.number_input("Price MA7", value=159.371647)
    price_ma2 = st.sidebar.number_input("Price MA2", value=159.371647)
    Solar_2da = st.sidebar.number_input("Solar 2da", value=920.903375)
    Wind_2da = st.sidebar.number_input("Wind 2da", value=7526.838)
    Coal = st.sidebar.number_input("Coal", value=13631)
    Gas = st.sidebar.number_input("Gas", value=37746)
    Nuclear = st.sidebar.number_input("Nuclear", value=33715)

    DOM_dem_7da = st.sidebar.number_input("DOM_dem_7da", value=13515)
    Hydro = st.sidebar.number_input("Hydro", value=1480)
    Oil = st.sidebar.number_input("Oil", value=221)

    return pd.DataFrame({
        "hour": [hour], "dayofweek": [dayofweek], "month": [month],
        "price_lag_1": [price_lag_1], "price_lag_2": [price_lag_2], "price_lag_7": [price_lag_7],
        "price_ma7": [price_ma7], "price_ma2": [price_ma2],
        "Solar_2da": [Solar_2da], "Wind_2da": [Wind_2da], "DOM_dem_7da": [DOM_dem_7da], "Oil": [Oil], "Hydro": [Hydro],
        "Coal": [Coal], "Gas": [Gas], "Nuclear": [Nuclear]
    })


# Input form
input_df = user_input()
print(input_df)
# Scale input data
input_scaled = scaler.transform(input_df)

# Make predictions
predictions = {quantile: model.predict(
    input_scaled)[0] for quantile, model in models.items()}

print(predictions)
# Display results
st.write("### Price Forecast")
st.write(f"10th Percentile (Lower Bound): {predictions['p10']:.2f}")
st.write(f"50th Percentile (Median): {predictions['p50']:.2f}")
st.write(f"90th Percentile (Upper Bound): {predictions['p90']:.2f}")

# Visualize the uncertainty
st.write("### Forecast Uncertainty Visualization")
fig, ax = plt.subplots()
ax.bar(["10th Percentile", "50th Percentile", "90th Percentile"],
       [predictions['p10'], predictions['p50'], predictions['p90']], color=['red', 'green', 'blue'])
ax.set_ylabel('Price')
ax.set_title('Price Forecast and Uncertainty')
st.pyplot(fig)
