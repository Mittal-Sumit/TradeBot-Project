import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\TradeBot Project\TRADEBOT_Market_Data.csv', parse_dates=['DateTime'])
print(df.head())
# Set DateTime as index if needed
df.set_index('DateTime', inplace=True)

# Resample to daily frequency and calculate required metrics
daily_data = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(daily_data.head())

daily_data.to_csv('file1.csv')