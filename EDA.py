import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
  # Assuming you have a module named EDA_functions.py
# Load the dataset
df = pd.read_csv(r'C:\Users\egang\OneDrive\Desktop\Sales forecasting\Sales_forecating-DS\Walmart.csv')  

# Basic info
print(df.shape)
print(df.columns)
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Convert date column to datetime with correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Sort by date
df = df.sort_values('Date')

# Add extra date features
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week

# Total sales over time
plt.figure(figsize=(12,6))
df.groupby('Date')['Weekly_Sales'].sum().plot()
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

