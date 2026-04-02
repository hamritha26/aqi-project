code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Urban AQI ML Dashboard")

# Load dataset
data = pd.read_csv("city_day.csv")

# Show dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Description
st.subheader("Dataset Description")
st.write(data.describe())

# Correlation heatmap
st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)

st.pyplot(fig)

# Model comparison (replace with your values if needed)
st.subheader("Model Comparison")
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "R2 Score": [0.72, 0.81, 0.90, 0.92]
})

st.write(results)
st.bar_chart(results.set_index("Model"))

# Conclusion
st.subheader("Conclusion")
st.write("Random Forest and XGBoost performed best.")
"""

with open("app.py", "w") as f:
    f.write(code)
