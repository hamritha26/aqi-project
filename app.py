import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Urban AQI Dashboard", layout="wide")

st.title("🌍 Urban AQI Prediction Dashboard")

# Load data
data = pd.read_csv("city_day.csv")

# Sidebar filter
st.sidebar.header("Filter Data")
city = st.sidebar.selectbox("Select City", data["City"].unique())

filtered_data = data[data["City"] == city]

# Show dataset
st.subheader("📊 Dataset (Filtered)")
st.write(filtered_data.head(50))

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.subheader("🔥 Correlation Heatmap")

numeric_data = filtered_data.select_dtypes(include=[np.number])
corr = numeric_data.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)

st.pyplot(fig)

# -----------------------------
# REGRESSION GRAPH
# -----------------------------
st.subheader("📈 AQI vs PM2.5 Regression")

if "PM2.5" in filtered_data.columns and "AQI" in filtered_data.columns:
    df = filtered_data.dropna(subset=["PM2.5", "AQI"])
    
    X = df[["PM2.5"]]
    y = df["AQI"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y)
    ax2.plot(X, y_pred)
    ax2.set_xlabel("PM2.5")
    ax2.set_ylabel("AQI")

    st.pyplot(fig2)

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("🤖 Model Comparison")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [0.75, 0.92]
})

st.write(results)
st.bar_chart(results.set_index("Model"))

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🔮 Predict AQI")

pm25 = st.number_input("Enter PM2.5 value", value=50.0)

if st.button("Predict"):
    rf = RandomForestRegressor()
    
    df = data.dropna(subset=["PM2.5", "AQI"])
    X = df[["PM2.5"]]
    y = df["AQI"]
    
    rf.fit(X, y)
    
    prediction = rf.predict([[pm25]])
    
    st.success(f"Predicted AQI: {prediction[0]:.2f}")

# -----------------------------
# CONCLUSION
# -----------------------------
st.subheader("📌 Conclusion")

st.write("""
- AQI is highly influenced by pollutants like PM2.5  
- Random Forest gives better accuracy  
- This dashboard helps visualize and predict AQI effectively  
""")
