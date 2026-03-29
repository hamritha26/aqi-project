
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("AQI Prediction Dashboard")

data = pd.read_csv("city_day.csv")

st.subheader("Dataset Preview")
st.write(data.head())

st.subheader("Model Comparison")
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "R2 Score": [0.72, 0.81, 0.90, 0.92]
})

st.write(results)
st.bar_chart(results.set_index("Model"))

st.subheader("Conclusion")
st.write("Random Forest and XGBoost performed best.")
