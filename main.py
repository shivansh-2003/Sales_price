import streamlit as st
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Product Sales Forecasting", layout="wide")

# Title of the app
st.title("Product Sales Forecasting with SARIMAX")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Input fields
product_id = st.number_input("Enter Product ID", min_value=0, step=1, format="%d")

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Functions for feature engineering, aggregation, and training
def feature_engineering_and_preprocessing(df_product):
    # Example feature engineering (replace with actual function)
    return df_product

def aggregate_weekly_sales(df_product):
    df_product['transaction_date'] = pd.to_datetime(df_product['transaction_date'])
    df_product.set_index('transaction_date', inplace=True)
    
    # Aggregating sales data to weekly
    weekly_sales = df_product['quantity'].resample('W').sum()
    
    # Aggregating exogenous variables to weekly and handling NaN/Inf
    exogenous_weekly = df_product[['discount_applied', 'unit_price', 'product_stock']].resample('W').mean()
    
    # Replace infinite values with NaN, then fill NaNs with suitable method
    exogenous_weekly = exogenous_weekly.replace([np.inf, -np.inf], np.nan)
    exogenous_weekly = exogenous_weekly.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill as needed

    return weekly_sales, exogenous_weekly


def train_sarimax_model(df, product_id, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    df_product = df[df['product_id'] == product_id]
    if df_product.empty:
        st.error(f"No data found for product ID {product_id}")
        return None, None, None

    df_product = feature_engineering_and_preprocessing(df_product)
    weekly_sales, exogenous_weekly = aggregate_weekly_sales(df_product)
    train_size = int(len(weekly_sales) * 0.8)
    train_sales, test_sales = weekly_sales[:train_size], weekly_sales[train_size:]
    train_exog, test_exog = exogenous_weekly[:train_size], exogenous_weekly[train_size:]
    
    model = SARIMAX(train_sales, exog=train_exog, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    
    return model_fit, test_sales, test_exog

def forecast_and_evaluate(model_fit, test_sales, test_exog, forecast_steps=12):
    forecast = model_fit.forecast(steps=forecast_steps, exog=test_exog[:forecast_steps])
    mae = mean_absolute_error(test_sales[:forecast_steps], forecast)
    rmse = np.sqrt(mean_squared_error(test_sales[:forecast_steps], forecast))
    mape = np.mean(np.abs((test_sales[:forecast_steps] - forecast) / test_sales[:forecast_steps])) * 100
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_sales.index, test_sales, label='Actual Sales', marker='o')
    ax.plot(test_sales.index[:forecast_steps], forecast, color='red', label='Forecasted Sales', marker='x')
    ax.set_title("Sales Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.legend()
    ax.grid()
    
    return forecast, {"MAE": mae, "RMSE": rmse, "MAPE": mape}, fig

# Run forecast and display results
if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    if st.button("Train Model and Forecast"):
        model_fit, test_sales, test_exog = train_sarimax_model(data, product_id)
        
        if model_fit is not None:
            forecast, metrics, fig = forecast_and_evaluate(model_fit, test_sales, test_exog)
            
            # Display metrics
            st.subheader("Evaluation Metrics")
            st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
            
            # Display plot
            st.pyplot(fig)
