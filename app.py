import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.title("🌍 Urban AQI Analysis & Prediction Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    data = pd.read_csv("city_day.csv")
    return data

data = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🏙 City Analysis",
    "📈 Correlation",
    "🤖 ML Models",
    "🔮 Prediction Simulator"
])

# ------------------ COMMON FEATURES ------------------
features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Toluene"]
target = "AQI"

df = data.dropna()

# ------------------ OVERVIEW ------------------
if section == "📊 Overview":
    st.header("📊 Dataset Overview")

    st.subheader("Sample Data")
    st.dataframe(data.head())

    st.subheader("City Distribution")
    st.bar_chart(data["City"].value_counts())

# ------------------ CITY ANALYSIS ------------------
elif section == "🏙 City Analysis":
    st.header("🏙 Single City Analysis")

    city = st.selectbox("Select City", data["City"].unique())

    city_data = data[data["City"] == city]

    st.subheader(f"📍 Data for {city}")
    st.dataframe(city_data.tail())

    st.subheader("📈 AQI Trend")
    fig, ax = plt.subplots()
    ax.plot(city_data["AQI"])
    ax.set_title(f"AQI Trend for {city}")
    st.pyplot(fig)

    st.subheader("🌫 Pollutant Levels")
    st.line_chart(city_data[features])

# ------------------ CORRELATION ------------------
elif section == "📈 Correlation":
    st.header("📈 Correlation Analysis")

    corr = df[features + [target]].corr()

    st.subheader("Correlation Matrix")
    st.dataframe(corr)

    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    st.pyplot(fig)

# ------------------ ML MODELS ------------------
elif section == "🤖 ML Models":
    st.header("🤖 Model Training & Evaluation")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    score = r2_score(y_test, preds)

    st.subheader("📊 Model Performance")
    st.success(f"R2 Score: {score:.2f}")

    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds)
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    st.pyplot(fig)

# ------------------ PREDICTION SIMULATOR ------------------
elif section == "🔮 Prediction Simulator":
    st.header("🔮 AQI Prediction Simulator")

    # Train model
    X = df[features]
    y = df[target]

    model = RandomForestRegressor()
    model.fit(X, y)

    # Select city
    city = st.selectbox("Select City", data["City"].unique())

    city_data = data[data["City"] == city].dropna()

    latest = city_data.iloc[-1]

    st.subheader("📍 Current Pollution Levels")
    st.write(latest[features])

    st.subheader("⚙️ Adjust Pollution Levels")

    input_data = {}

    for col in features:
        input_data[col] = st.slider(
            f"{col}",
            0.0,
            float(latest[col] * 2),
            float(latest[col])
        )

    input_df = pd.DataFrame([input_data])

    predicted_aqi = model.predict(input_df)[0]
    original_aqi = latest["AQI"]

    st.subheader("📊 AQI Prediction Result")

    st.metric(
        label="Predicted AQI",
        value=round(predicted_aqi, 2),
        delta=round(predicted_aqi - original_aqi, 2)
    )

    if predicted_aqi > original_aqi:
        st.error("⚠️ Air Quality Worsened")
    else:
        st.success("✅ Air Quality Improved")

    st.subheader("📈 Impact Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Original AQI", "Predicted AQI"], [original_aqi, predicted_aqi])
    st.pyplot(fig)
