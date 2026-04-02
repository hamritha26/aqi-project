import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AQI Data Science Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("city_day.csv")

data = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🌍 Navigation")

page = st.sidebar.radio("Select Module", [
    "🏠 Overview",
    "📊 EDA",
    "📍 Single City",
    "🌆 Multi City",
    "🤖 ML Training",
    "📈 Model Evaluation",
    "🔮 Prediction"
])

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "🏠 Overview":
    st.title("🌍 AQI Data Science Dashboard")

    st.write("### Dataset Summary")
    st.write(data.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Cities", data["City"].nunique())
    col3.metric("Avg AQI", round(data["AQI"].mean(), 2))

# -----------------------------
# EDA
# -----------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset")
    st.write(data)

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    st.subheader("Correlation Heatmap")
    numeric = data.select_dtypes(include=np.number)
    corr = numeric.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    st.pyplot(fig)

# -----------------------------
# SINGLE CITY
# -----------------------------
elif page == "📍 Single City":
    city = st.selectbox("Select City", data["City"].unique())
    df = data[data["City"] == city]

    st.title(f"📍 {city} Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg AQI", round(df["AQI"].mean(), 2))
    col2.metric("Max AQI", round(df["AQI"].max(), 2))
    col3.metric("Min AQI", round(df["AQI"].min(), 2))

    st.line_chart(df["AQI"])

# -----------------------------
# MULTI CITY
# -----------------------------
elif page == "🌆 Multi City":
    cities = st.multiselect("Select Cities", data["City"].unique())

    if cities:
        df = data[data["City"].isin(cities)]
        st.bar_chart(df.groupby("City")["AQI"].mean())

# -----------------------------
# ML TRAINING
# -----------------------------
elif page == "🤖 ML Training":

    st.title("🤖 Train Models")

    df = data.dropna(subset=["PM2.5", "AQI"])
    X = df[["PM2.5"]]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = st.selectbox("Choose Model", [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ])

    if st.button("Train Model"):

        if model_choice == "Linear Regression":
            model = LinearRegression()

        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()

        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        st.success(f"{model_choice} R2 Score: {round(score, 3)}")

        st.session_state["model"] = model

# -----------------------------
# MODEL EVALUATION
# -----------------------------
elif page == "📈 Model Evaluation":

    st.title("📈 Model Evaluation")

    if "model" in st.session_state:

        model = st.session_state["model"]

        df = data.dropna(subset=["PM2.5", "AQI"])
        X = df[["PM2.5"]]
        y = df["AQI"]

        preds = model.predict(X)

        fig, ax = plt.subplots()
        ax.scatter(y, preds)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        st.pyplot(fig)

    else:
        st.warning("Train a model first")

# -----------------------------
# PREDICTION
# -----------------------------
elif page == "🔮 Prediction":

    st.title("🔮 AQI Prediction")

    pm25 = st.slider("PM2.5", 0, 500, 50)

    if "model" in st.session_state:
        model = st.session_state["model"]
        pred = model.predict([[pm25]])

        st.metric("Predicted AQI", round(pred[0], 2))
    else:
        st.warning("Train model first")
