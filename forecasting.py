from EDA import df,plt
import pandas as pd

top_stores = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head()
print(top_stores)

df['Date'] = pd.to_datetime(df['Date'])  # make sure date is datetime
sales_by_date = df.groupby('Date')['Weekly_Sales'].sum()

plt.figure(figsize=(14, 6))
sales_by_date.plot()
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
