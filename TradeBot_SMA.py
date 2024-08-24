import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\TradeBot Project\TRADEBOT_Market_Data.csv', parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)

# Calculate DailyMetrics
daily_data = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(daily_data.head())

# Calculate Daily Return Percentage
daily_data['Daily Return'] = (daily_data['Close']-daily_data['Open'])/daily_data['Open'] * 100

# Calculate Moving Averages
daily_data['SMA_7'] = daily_data['Close'].rolling(window=7).mean()
daily_data['SMA_14'] = daily_data['Close'].rolling(window=14).mean()
daily_data['SMA_30'] = daily_data['Close'].rolling(window=30).mean()

print(daily_data.head(30))  

# Initialize signals DataFrame
signals = pd.DataFrame(index=daily_data.index)
signals['Close'] = daily_data['Close']
signals['SMA_30'] = daily_data['SMA_30']
signals['Signal'] = 0  # Default to no signal

# Generate Buy/Sell signals
signals['Signal'][daily_data['Close'] > daily_data['SMA_30']] = 1  
signals['Signal'][daily_data['Close'] < daily_data['SMA_30']] = 0  

# Calculate positions
signals['Position'] = signals['Signal'].diff()

print(signals.head(60))  # Print more rows to see signals and positions
# Extract entry and exit points
entries = signals[signals['Position'] == 1]
exits = signals[signals['Position'] == -1]

print("Entries:")
print(entries)
print("\nExits:")
print(exits)

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_data['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(daily_data['SMA_30'], label='30-day SMA', color='orange', alpha=0.75)

# Plot entry and exit points
plt.plot(entries.index, daily_data['Close'][entries.index], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(exits.index, daily_data['Close'][exits.index], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Stock Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()







