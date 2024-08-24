import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data and set the datetime index
df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\TradeBot Project\TRADEBOT_Market_Data.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Calculate Daily Metrics
daily_data = df.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Calculate Daily Return Percentage
daily_data['Daily Return'] = (daily_data['close']-daily_data['open'])/daily_data['open'] * 100

# Calculate Moving Averages
daily_data['SMA_7'] = daily_data['close'].rolling(window=7).mean()
daily_data['SMA_14'] = daily_data['close'].rolling(window=14).mean()
daily_data['SMA_30'] = daily_data['close'].rolling(window=30).mean()
print(daily_data.head(30)) 
# Initialize signals DataFrame
signals = pd.DataFrame(index=daily_data.index)
signals['close'] = daily_data['close']
signals['SMA_30'] = daily_data['SMA_30']
signals['Signal'] = 0  # Default to no signal

# Generate Buy/Sell signals
signals['Signal'][daily_data['close'] > daily_data['SMA_30']] = 1  
signals['Signal'][daily_data['close'] < daily_data['SMA_30']] = 0  

# Calculate positions
signals['Position'] = signals['Signal'].diff()

# Extract entry and exit points
entries = signals[signals['Position'] == 1]
exits = signals[signals['Position'] == -1]

# Backtesting
initial_capital = 100000.0
positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions['Stock'] = signals['Signal']  # Number of shares held (1 or 0 for long-only strategy)

# Calculate portfolio value
portfolio = positions.multiply(daily_data['close'], axis=0)
pos_diff = positions.diff()

portfolio['holdings'] = (positions.multiply(daily_data['close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(daily_data['close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_data['close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(daily_data['SMA_30'], label='30-day SMA', color='orange', alpha=0.75)

# Plot entry and exit points
plt.plot(entries.index, daily_data['close'][entries.index], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(exits.index, daily_data['close'][exits.index], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Stock Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot portfolio value
plt.figure(figsize=(14, 7))
plt.plot(portfolio['total'], label='Portfolio value', color='blue', alpha=0.5)
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Print performance metrics
total_return = portfolio['total'][-1] / initial_capital - 1
annualized_return = ((1 + total_return) ** (365.0 / len(portfolio)) - 1)
annualized_volatility = portfolio['returns'].std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.03) / annualized_volatility  # Assuming risk-free rate of 3%

print(f"Total Return: {total_return:.2f}")
print(f"Annualized Return: {annualized_return:.2f}")
print(f"Annualized Volatility: {annualized_volatility:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
