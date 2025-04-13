import pandas as pd
from prophet import Prophet
from EDA import df, plt

# Group total weekly sales (across all stores)
sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Rename columns for Prophet
sales.columns = ['ds', 'y']  # Prophet needs 'ds' = date, 'y' = target

# Check the data
print(sales.head())


# Initialize Prophet model
model = Prophet()

# Fit the model
model.fit(sales)

# Forecast for the next 12 weeks
future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)

# View forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
