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
tab1, tab2, tab3, tab4, tab5, tab6 , tab7= st.tabs([
    "🌍 Overview",
    "🏙️ City Analysis",
    "📊 Insights",
    "🧪 Prediction",
    "🌆 Compare Cities",
    "🤖 ML Models",
    "🧠 SHAP"
    
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

# ================= TAB 4: INSIGHTS =================
with tab3:
    st.header("📊 Data Insights & Analysis")

    # -------- BASIC STATS --------
    st.subheader("📈 Dataset Overview")

    st.write("Total Records:", len(data))
    st.write("Total Cities:", data["City"].nunique())

    # -------- AQI DISTRIBUTION --------
    st.subheader("📊 AQI Distribution")

    fig, ax = plt.subplots()
    data["AQI"].hist(ax=ax)
    ax.set_title("AQI Distribution")
    st.pyplot(fig)

    # -------- CORRELATION --------
    st.subheader("🔗 Correlation Between Pollutants")

    corr = data[features + ["AQI"]].corr()

    fig, ax = plt.subplots()
    im = ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)

    st.pyplot(fig)

    # -------- TOP POLLUTED CITIES --------
    st.subheader("🏭 Most Polluted Cities")

    city_aqi = data.groupby("City")["AQI"].mean().sort_values(ascending=False)
    st.bar_chart(city_aqi.head(10))

    # -------- CLEANEST CITIES --------
    st.subheader("🌿 Cleanest Cities")

    st.bar_chart(city_aqi.tail(10))

    # -------- INSIGHT TEXT --------
    st.subheader("🧠 Key Insights")

    worst_city = city_aqi.idxmax()
    best_city = city_aqi.idxmin()

    st.error(f"⚠️ {worst_city} is the most polluted city")
    st.success(f"✅ {best_city} has the best air quality")

    st.info("PM2.5 and PM10 usually have the highest impact on AQI.")

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

        # -------- AQI COMPARISON --------
        st.subheader("📊 AQI Comparison")

        fig, ax = plt.subplots()
        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        for i, city in enumerate(selected_cities):
            city_df = compare_data[compare_data["City"] == city]
            ax.plot(city_df["AQI"], label=city, color=colors[i % len(colors)])

        ax.set_title("AQI Comparison Between Cities")
        ax.legend()
        st.pyplot(fig)

        # -------- AVERAGE AQI --------
        st.subheader("📈 Average AQI Comparison")

        avg_aqi = compare_data.groupby("City")["AQI"].mean()
        st.bar_chart(avg_aqi)

        # -------- AQI CATEGORY FUNCTION --------
        def aqi_category(aqi):
            if aqi <= 50:
                return "🟢 Good"
            elif aqi <= 100:
                return "🟡 Satisfactory"
            elif aqi <= 200:
                return "🟠 Moderate"
            elif aqi <= 300:
                return "🔴 Poor"
            elif aqi <= 400:
                return "🟣 Very Poor"
            else:
                return "⚫ Severe"

        # -------- SHOW CATEGORY --------
        st.subheader("🌫 AQI Category by City")

        for city in avg_aqi.index:
            st.info(f"{city}: {aqi_category(avg_aqi[city])}")

        # -------- INSIGHTS --------
        st.subheader("🧠 Insights")

        best_city = avg_aqi.idxmin()
        worst_city = avg_aqi.idxmax()

        st.success(f"✅ {best_city} has the best air quality")
        st.error(f"⚠️ {worst_city} has the worst air quality")

        # -------- RANKING --------
        st.subheader("🏆 City Ranking")

        ranking = avg_aqi.sort_values()
        st.dataframe(ranking)

        # -------- POLLUTANT COMPARISON --------
        st.subheader("🌫 Pollutant Comparison")

        pollutant = st.selectbox("Select Pollutant", features)

        fig2, ax2 = plt.subplots()

        for i, city in enumerate(selected_cities):
            city_df = compare_data[compare_data["City"] == city]
            ax2.plot(city_df[pollutant], label=city, color=colors[i % len(colors)])

        ax2.set_title(f"{pollutant} Comparison")
        ax2.legend()
        st.pyplot(fig2)

        # -------- EXTRA INSIGHT --------
        avg_pollutant = compare_data.groupby("City")[pollutant].mean()
        highest = avg_pollutant.idxmax()

        st.warning(f"⚠️ {highest} has highest {pollutant} levels")
        # ================= TAB 6: ML MODELS =================
with tab6:
    st.header("🤖 Machine Learning Models & Comparison")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd

    # -------- DATA --------
    X = data[features].fillna(0)
    y = data["AQI"].fillna(0)

    # -------- SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("📊 Data Split")
    st.write(f"Training: {len(X_train)}")
    st.write(f"Testing: {len(X_test)}")

    # -------- MODELS --------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=50)
    }

    results = {}

    st.subheader("⚙️ Training & Evaluation")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        st.write(f"### {name}")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        st.markdown("---")

    # -------- COMPARISON --------
    st.subheader("🏆 Model Comparison")

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df)

    st.bar_chart(results_df["R2"])

    best_model = results_df["R2"].idxmax()
    st.success(f"✅ Best Model: {best_model}")

    # -------- FEATURE IMPORTANCE --------
    st.subheader("📌 Feature Importance (Random Forest)")

    rf_model = models["Random Forest"]
    importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))
    # -------- TAB 7: SHAP --------
with tab7:
    st.subheader("🧠 SHAP Explainability")

    st.write("Understand how each pollutant affects AQI predictions")

    import shap

    # Use Random Forest for SHAP
    model = rf_model   # make sure rf_model is already trained

    X_sample = X_test.sample(100)  # small sample for speed

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        st.write("### 🌍 Global Feature Importance")
        fig1 = plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig1)

        st.write("### 🔍 Single Prediction Explanation")
        index = st.slider("Select Sample Index", 0, len(X_sample)-1, 0)

        fig2 = plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[index],
                base_values=explainer.expected_value,
                data=X_sample.iloc[index]
            )
        )
        st.pyplot(fig2)

    except Exception as e:
        st.error("SHAP Error: " + str(e))



  

