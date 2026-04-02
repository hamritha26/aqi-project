import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.title("🌍 Urban AQI Analysis & Prediction Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("city_day.csv")

data = load_data()

# ---------------- FEATURES ----------------
features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Toluene"]
target = "AQI"

df = data.dropna(subset=features + [target])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏙 City Analysis",
    "🤖 ML Model",
    "🔮 Prediction",
    "🌆 City Comparison"
])
# ================= TAB 1: OVERVIEW =================
with tab1:
    st.header("📊 Dataset Overview")

    st.subheader("Sample Data")
    st.dataframe(data.head())

    st.subheader("City Distribution")
    st.bar_chart(data["City"].value_counts())

# ================= TAB 2: CITY ANALYSIS =================
with tab2:
    st.header("🏙 City Analysis")

    city = st.selectbox("Select City", data["City"].unique())

    city_data = data[data["City"] == city]

    if city_data.empty:
        st.error("No data available for this city")
    else:
        st.subheader(f"📍 Data for {city}")
        st.dataframe(city_data.tail())

        st.subheader("📈 AQI Trend")
        st.write("This graph shows AQI changes over time.")
        st.line_chart(city_data["AQI"])

        with st.expander("🌫 Show Pollutant Levels"):
            st.line_chart(city_data[features])

# ================= TAB 3: ML MODEL =================
with tab3:
    st.header("🤖 Model Training & Evaluation")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    st.subheader("📊 Model Performance")
    st.success(f"R² Score: {score:.2f}")

    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds)
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    st.pyplot(fig)

# ================= TAB 4: PREDICTION =================
with tab4:
    st.header("🔮 AQI Prediction Simulator")

    st.write("Adjust pollutant levels to predict AQI")

    # Train model on full data
    model = RandomForestRegressor()
    model.fit(df[features], df[target])

    city = st.selectbox("Select City for Prediction", data["City"].unique())

    city_data = data[data["City"] == city]
    city_data = city_data.dropna(subset=features + [target])

    if city_data.empty:
        st.error("No valid data available for this city")
    else:
        city_data = city_data.sort_values(by="Date")
        latest = city_data.iloc[-1]

        st.subheader("📍 Current Pollution Levels")
        st.write(latest[features])

        st.subheader("⚙️ Adjust Pollution Levels")

        input_data = {}

        for col in features:
            input_data[col] = st.slider(
                col,
                0.0,
                float(latest[col] * 2),
                float(latest[col])
            )

        input_df = pd.DataFrame([input_data])

        predicted_aqi = model.predict(input_df)[0]
        original_aqi = latest["AQI"]

        st.subheader("📊 Prediction Result")

        st.metric(
            label="Predicted AQI",
            value=round(predicted_aqi, 2),
            delta=round(predicted_aqi - original_aqi, 2)
        )

        if predicted_aqi > original_aqi:
            st.error("⚠️ Air Quality Worsened")
        else:
            st.success("✅ Air Quality Improved")

        st.subheader("📊 Comparison")

        fig, ax = plt.subplots()
        ax.bar(["Original AQI", "Predicted AQI"], [original_aqi, predicted_aqi])
        st.pyplot(fig)
        # ================= TAB 5: CITY COMPARISON =================
with tab5:
    st.header("🌆 City Comparison Analysis")

    selected_cities = st.multiselect(
        "Select Cities to Compare",
        data["City"].unique()
    )

    if len(selected_cities) < 2:
        st.warning("Please select at least 2 cities")
    else:
        compare_data = data[data["City"].isin(selected_cities)]

        st.subheader("📊 AQI Comparison")

        fig, ax = plt.subplots()

        colors = ["red", "blue", "green", "orange", "purple", "brown"]

for i, city in enumerate(selected_cities):
    city_df = compare_data[compare_data["City"] == city]
    ax.plot(city_df["AQI"], label=city, color=colors[i % len(colors)])

        ax.set_title("AQI Comparison Between Cities")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📈 Average AQI Comparison")

        avg_aqi = compare_data.groupby("City")["AQI"].mean()
        st.bar_chart(avg_aqi)

        with st.expander("🌫 Compare Pollutants"):
            pollutant = st.selectbox("Select Pollutant", features)

            fig, ax = plt.subplots()

           for i, city in enumerate(selected_cities):
    city_df = compare_data[compare_data["City"] == city]
    ax.plot(city_df[pollutant], label=city, color=colors[i % len(colors)])

            ax.set_title(f"{pollutant} Comparison")
            ax.legend()
            st.pyplot(fig)
