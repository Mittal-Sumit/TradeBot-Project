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

# Calculate Daily Return Percentage
daily_data['Daily Return'] = daily_data['Close'].pct_change() * 100

# Calculate Moving Averages
daily_data['SMA_7'] = daily_data['Close'].rolling(window=7).mean()
daily_data['SMA_14'] = daily_data['Close'].rolling(window=14).mean()
daily_data['SMA_30'] = daily_data['Close'].rolling(window=30).mean()

print(daily_data.head(30))  # Print more rows to see moving averages

# Initialize signals DataFrame
signals = pd.DataFrame(index=daily_data.index)
signals['Close'] = daily_data['Close']
signals['SMA_30'] = daily_data['SMA_30']
signals['Signal'] = 0  # Default to no signal

# Generate Buy/Sell signals
signals['Signal'][daily_data['Close'] > daily_data['SMA_30']] = 1  # Buy signal
signals['Signal'][daily_data['Close'] < daily_data['SMA_30']] = -1  # Sell signal

# Calculate positions
signals['Position'] = signals['Signal'].diff()

print(signals.head(60))  # Print more rows to see signals and positions
# Extract entry and exit points
entries = signals[signals['Position'] == 1]
exits = signals[signals['Position'] == -1]

# Calculate percentage increase
total_return = (daily_data['Close'][-1] / daily_data['Close'][0] - 1) * 100

print("Entries:")
print(entries)
print("\nExits:")
print(exits)
print(f"\nTotal percentage increase over the period: {total_return:.2f}%")

daily_data.to_csv('file1.csv')