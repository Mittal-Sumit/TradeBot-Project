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

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window, no_of_std):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    return rolling_mean, upper_band, lower_band

daily_data['SMA_20'], daily_data['Upper_Band'], daily_data['Lower_Band'] = calculate_bollinger_bands(daily_data, 20, 2)

# Create signals DataFrame for Bollinger Bands strategy
bb_signals = pd.DataFrame(index=daily_data.index)
bb_signals['Close'] = daily_data['Close']
bb_signals['Upper_Band'] = daily_data['Upper_Band']
bb_signals['Lower_Band'] = daily_data['Lower_Band']
bb_signals['Signal'] = 0
bb_signals['Signal'][daily_data['Close'] < daily_data['Lower_Band']] = 1  # Buy signal
bb_signals['Signal'][daily_data['Close'] > daily_data['Upper_Band']] = -1  # Sell signal
bb_signals['Position'] = bb_signals['Signal'].diff()

# Backtesting
initial_capital = 100000.0
positions = pd.DataFrame(index=bb_signals.index).fillna(0.0)
positions['Stock'] = bb_signals['Signal']  # Number of shares held (1 or 0 for long-only strategy)

# Calculate portfolio value
portfolio = positions.multiply(daily_data['Close'], axis=0)
pos_diff = positions.diff()

portfolio['holdings'] = (positions.multiply(daily_data['Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(daily_data['Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Extract entry and exit points
entries = bb_signals[bb_signals['Position'] == 1]
exits = bb_signals[bb_signals['Position'] == -1]

# Limit the backtest to the last 6 months
six_months_ago = daily_data.index[-1] - pd.DateOffset(months=6)
portfolio_six_months = portfolio[portfolio.index >= six_months_ago]
bb_signals_six_months = bb_signals[bb_signals.index >= six_months_ago]

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_data['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(daily_data['Upper_Band'], label='Upper Band', color='red', alpha=0.75)
plt.plot(daily_data['Lower_Band'], label='Lower Band', color='green', alpha=0.75)

# Plot entry and exit points
plt.plot(entries.index, daily_data['Close'][entries.index], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(exits.index, daily_data['Close'][exits.index], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Stock Price with Bollinger Bands Buy and Sell Signals')
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
print("\nBollinger Bands Strategy Report:")
print("Entry Points (last 6 months):")
print(entries[entries.index >= six_months_ago][['Close', 'Lower_Band']])
print("\nExit Points (last 6 months):")
print(exits[exits.index >= six_months_ago][['Close', 'Upper_Band']])
total_percentage_increase_bb = portfolio_six_months['total'].iloc[-1] / initial_capital * 100 - 100
print(f"Total Percentage Increase by Bollinger Bands Strategy (last 6 months): {total_percentage_increase_bb:.2f}%")

# Performance Metrics for last 6 months
annualized_return = ((1 + portfolio_six_months['total'].iloc[-1] / initial_capital - 1) ** (365.0 / len(portfolio_six_months)) - 1)
annualized_volatility = portfolio_six_months['returns'].std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.03) / annualized_volatility  # Assuming risk-free rate of 3%

print(f"Annualized Return (last 6 months): {annualized_return:.2f}")
print(f"Annualized Volatility (last 6 months): {annualized_volatility:.2f}")
print(f"Sharpe Ratio (last 6 months): {sharpe_ratio:.2f}")
