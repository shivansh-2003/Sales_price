import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame


df = pd.read_csv("1.csv")

# Convert 'transaction_date' to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

def feature_engineering_and_preprocessing(df):
    df.loc[:, 'week_of_year'] = df['transaction_date'].dt.isocalendar().week
    df.loc[:, 'year'] = df['transaction_date'].dt.year

    features = ['discount_applied', 'unit_price', 'product_stock', 'product_category', 'day_of_week', 'season', 'week_of_year', 'year', 'product_id']
    df = df[['transaction_date', 'quantity'] + features]

    categorical_features = ['product_category', 'day_of_week', 'season']
    numerical_features = ['discount_applied', 'unit_price', 'product_stock']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    exogenous_vars = df.drop(columns=['transaction_date', 'quantity'])
    exogenous_preprocessed = preprocessor.fit_transform(exogenous_vars)

    return df, exogenous_preprocessed, preprocessor

def aggregate_weekly_sales(df):
    df.set_index('transaction_date', inplace=True)

    # Resample and sum for weekly sales
    weekly_sales = df['quantity'].resample('W').sum()

    # Resample and mean for exogenous features
    exogenous_features = df[['discount_applied', 'unit_price', 'product_stock', 'product_category', 'day_of_week', 'season', 'product_id']]
    exogenous_weekly = exogenous_features.resample('W').mean()

    # Drop any rows with NaN values
    weekly_sales = weekly_sales.dropna()
    exogenous_weekly = exogenous_weekly.dropna()

    # Ensure that the indices are aligned
    aligned = weekly_sales.index.intersection(exogenous_weekly.index)
    return weekly_sales.loc[aligned], exogenous_weekly.loc[aligned]

def train_sarimax_model(df, product_id, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), forecast_steps=12):
    df_product = df[df['product_id'] == product_id]

    if df_product.empty:
        raise ValueError(f"No data found for product ID {product_id}")

    df_product, exogenous_preprocessed, preprocessor = feature_engineering_and_preprocessing(df_product)

    weekly_sales, exogenous_weekly = aggregate_weekly_sales(df_product)

    # Align indices before fitting the model
    if len(weekly_sales) == 0 or len(exogenous_weekly) == 0:
        raise ValueError(f"No valid sales or exogenous data available for product ID {product_id}")

    # Train the SARIMAX model
    model = SARIMAX(weekly_sales, exog=exogenous_weekly, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=forecast_steps, exog=exogenous_weekly[-forecast_steps:])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_sales.index, weekly_sales, label='Historical Sales', marker='o')
    plt.plot(pd.date_range(weekly_sales.index[-1], periods=forecast_steps + 1, freq='W')[1:], forecast, color='red', label='Forecasted Sales', marker='x')
    plt.title(f'Sales Forecast for Product ID {product_id} - Next {forecast_steps} Weeks')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.grid()
    plt.show()

    return forecast

# Train and forecast for all unique product IDs
unique_product_ids = df['product_id'].unique()

for product_id in unique_product_ids:
    print(f"Training model for Product ID: {product_id}")
    try:
        forecast = train_sarimax_model(df, product_id, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), forecast_steps=12)
        print(f"Forecast for Product ID {product_id}: {forecast.values}\n")
    except Exception as e:
        print(f"An error occurred for Product ID {product_id}: {e}\n")