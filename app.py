import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.preprocess import load_and_preprocess_data
from scripts.train import train_model
from scripts.prediction import predict_and_evaluate, save_predictions
from scripts.visualization import plot_results

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

# Load and preprocess
st.markdown("### 📂 Load and Preprocess Data")
data_path = "data/normalised.csv"

x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
data = pd.read_csv(data_path).drop(columns=["id", "date", "zipcode"], errors='ignore')

# Train
st.markdown("### 🔧 Train the Model")
model = train_model(x_train, y_train)

# Predict & Evaluate
y_pred, mse, r2 = predict_and_evaluate(model, x_test, y_test)

st.markdown("### 📊 Model Evaluation Metrics")
st.write(f"✅ **Mean Squared Error (MSE):** `{mse:.2f}`")
st.write(f"✅ **R-squared Score (R²):** `{r2:.4f}`")

# Save predictions
os.makedirs("results", exist_ok=True)
save_path = os.path.join("results", "predictions.csv")
save_predictions(y_test, y_pred, save_path)

# Show actual vs predicted plot
st.markdown("### 📈 Actual vs Predicted Prices")
plot_path = os.path.join("results", "actual_vs_predicted.png")
plot_results(y_test, y_pred, plot_path)
st.image(plot_path, caption="Actual vs Predicted Prices")

# Show feature-wise analysis
st.markdown("### 📊 Feature-wise Price Analysis")

X = data.drop(columns=["price"])
feature_col = st.selectbox("Select a feature for analysis:", options=X.columns)

# Average price per category
avg_prices = data.groupby(feature_col)["price"].mean().reset_index()
avg_prices = avg_prices.sort_values("price", ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(avg_prices[feature_col].astype(str), avg_prices["price"], color='lightblue')
ax2.set_title(f"Average Price by {feature_col}")
ax2.set_xlabel(feature_col)
ax2.set_ylabel("Average Price")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Boxplot
st.markdown("### 📦 Boxplot: Price Distribution by Feature")
max_categories = 10
unique_vals = data[feature_col].nunique()

if unique_vals > max_categories:
    st.warning(f"Too many categories in '{feature_col}', showing top {max_categories} most frequent.")
    top_vals = data[feature_col].value_counts().nlargest(max_categories).index
    filtered = data[data[feature_col].isin(top_vals)]
else:
    filtered = data

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered, x=feature_col, y="price", ax=ax3)
ax3.set_title(f"Price Distribution by {feature_col}")
plt.xticks(rotation=45)
st.pyplot(fig3)

st.success("✅ All graphs and predictions rendered successfully!")
