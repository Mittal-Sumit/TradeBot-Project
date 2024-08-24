import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\TradeBot Project\TRADEBOT_Market_Data.csv', parse_dates=['DateTime'])

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Check for duplicates
duplicates = df.duplicated().sum()
print("Duplicates:\n", duplicates)

# Remove duplicates if any
df = df.drop_duplicates()

# Ensure 'DateTime' is in the correct format and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Forward fill missing data points to maintain continuity
df = df.asfreq('D', method='pad')

# Validate date range consistency
print("Date range:\n", df.index.min(), " to ", df.index.max())

# Plot closing prices to visually inspect
df['Close'].plot(title='Closing Prices')
plt.show()

# Basic statistics to identify potential outliers
print("Close price statistics:\n", df['Close'].describe())

# Use adjusted close prices if available
# df['Adjusted_Close'] = df['Close']  # Example placeholder for adjusted prices

# Final cleaned dataset
print("Cleaned dataset:\n", df.head())

# Ensure there are no remaining missing values
assert df.isnull().sum().sum() == 0, "There are still missing values in the dataset."
