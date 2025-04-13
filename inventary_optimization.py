from sales_forecasting_prophet import forecast
from EDA import df, plt

# Assume forecast is already available from Prophet model
# Add a 10% safety stock buffer for inventory
forecast['recommended_stock'] = forecast['yhat'] * 1.10

# Print first few rows to check the stock recommendations
print(forecast[['ds', 'yhat', 'recommended_stock']].head())

plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='blue')
plt.plot(forecast['ds'], forecast['recommended_stock'], label='Recommended Inventory', color='green', linestyle='--')
plt.title("Sales Forecast vs Recommended Inventory (Next 12 Weeks)")
plt.xlabel("Date")
plt.ylabel("Sales / Inventory")
plt.legend()
plt.grid(True)
plt.show()


def calculate_inventory(forecast, safety_margin=0.10):
    """
    Calculate recommended inventory based on forecasted sales and safety margin
    """
    forecast['recommended_stock'] = forecast['yhat'] * (1 + safety_margin)
    return forecast[['ds', 'yhat', 'recommended_stock']]

# Apply function to get inventory recommendations
inventory = calculate_inventory(forecast)

# Print the updated inventory recommendations
print(inventory.head())

