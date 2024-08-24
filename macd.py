import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file with parse_dates parameter
df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\TradeBot Project\TRADEBOT_Market_Data.csv', parse_dates=['DateTime'])

# Set 'DateTime' as the index
df.set_index('DateTime', inplace=True)

# Aggregate data to daily frequency
daily_data = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

daily_data['MACD'], daily_data['Signal_Line'] = calculate_macd(daily_data)

# Create signals DataFrame for MACD strategy
macd_signals = pd.DataFrame(index=daily_data.index)
macd_signals['Close'] = daily_data['Close']
macd_signals['MACD'] = daily_data['MACD']
macd_signals['Signal_Line'] = daily_data['Signal_Line']
macd_signals['Signal'] = 0
macd_signals['Signal'][daily_data['MACD'] > daily_data['Signal_Line']] = 1  # Buy signal
macd_signals['Signal'][daily_data['MACD'] < daily_data['Signal_Line']] = -1  # Sell signal
macd_signals['Position'] = macd_signals['Signal'].diff()

# Backtesting
initial_capital = 100000.0
positions = pd.DataFrame(index=macd_signals.index).fillna(0.0)
positions['Stock'] = macd_signals['Signal']  # Number of shares held (1 or 0 for long-only strategy)

# Calculate portfolio value
portfolio = positions.multiply(daily_data['Close'], axis=0)
pos_diff = positions.diff()

portfolio['holdings'] = (positions.multiply(daily_data['Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(daily_data['Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Extract entry and exit points
entries = macd_signals[macd_signals['Position'] == 1]
exits = macd_signals[macd_signals['Position'] == -1]

# Limit the backtest to the last 6 months
six_months_ago = daily_data.index[-1] - pd.DateOffset(months=6)
portfolio_six_months = portfolio[portfolio.index >= six_months_ago]
macd_signals_six_months = macd_signals[macd_signals.index >= six_months_ago]

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_data['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(daily_data['MACD'], label='MACD', color='red', alpha=0.75)
plt.plot(daily_data['Signal_Line'], label='Signal Line', color='green', alpha=0.75)

# Plot entry and exit points
plt.plot(entries.index, daily_data['Close'][entries.index], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(exits.index, daily_data['Close'][exits.index], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Stock Price with MACD Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot portfolio value
plt.figure(figsize=(14, 7))
plt.plot(portfolio_six_months['total'], label='Portfolio value', color='blue', alpha=0.5)
plt.title('Portfolio Value Over Last 6 Months')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Generate report
print("\nMACD Strategy Report:")
print("Entry Points (last 6 months):")
print(entries[entries.index >= six_months_ago][['Close', 'MACD', 'Signal_Line']])
print("\nExit Points (last 6 months):")
print(exits[exits.index >= six_months_ago][['Close', 'MACD', 'Signal_Line']])
total_percentage_increase_macd = portfolio_six_months['total'].iloc[-1] / initial_capital * 100 - 100
print(f"Total Percentage Increase by MACD Strategy (last 6 months): {total_percentage_increase_macd:.2f}%")

# Performance Metrics for last 6 months
annualized_return = ((1 + portfolio_six_months['total'].iloc[-1] / initial_capital - 1) ** (365.0 / len(portfolio_six_months)) - 1)
annualized_volatility = portfolio_six_months['returns'].std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.03) / annualized_volatility  # Assuming risk-free rate of 3%

print(f"Annualized Return (last 6 months): {annualized_return:.2f}")
print(f"Annualized Volatility (last 6 months): {annualized_volatility:.2f}")
print(f"Sharpe Ratio (last 6 months): {sharpe_ratio:.2f}")
