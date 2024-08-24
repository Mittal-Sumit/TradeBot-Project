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

# Calculate daily returns
daily_data['Return'] = daily_data['Close'].pct_change()

# Calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

daily_data['RSI'] = calculate_rsi(daily_data, 14)

# Create signals DataFrame for RSI strategy
rsi_signals = pd.DataFrame(index=daily_data.index)
rsi_signals['Close'] = daily_data['Close']
rsi_signals['RSI'] = daily_data['RSI']
rsi_signals['Signal'] = 0
rsi_signals['Signal'][daily_data['RSI'] < 30] = 1  # Buy signal
rsi_signals['Signal'][daily_data['RSI'] > 70] = -1  # Sell signal
rsi_signals['Position'] = rsi_signals['Signal'].diff()

# Calculate strategy returns
rsi_signals['Strategy_Return'] = daily_data['Return'] * rsi_signals['Signal'].shift(1)
rsi_signals['Cumulative_Strategy_Return'] = (1 + rsi_signals['Strategy_Return']).cumprod() - 1
rsi_signals['Cumulative_Market_Return'] = (1 + daily_data['Return']).cumprod() - 1

# Print RSI signals DataFrame
print("\nRSI Signals DataFrame:")
print(rsi_signals)

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_data['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(daily_data['RSI'], label='RSI', color='orange', alpha=0.75)

# Plot entry and exit points
plt.plot(rsi_signals[rsi_signals['Position'] == 1].index, daily_data['Close'][rsi_signals['Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(rsi_signals[rsi_signals['Position'] == -1].index, daily_data['Close'][rsi_signals['Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Stock Price with RSI Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(rsi_signals['Cumulative_Strategy_Return'], label='Cumulative Strategy Return', color='blue', alpha=0.5)
plt.plot(rsi_signals['Cumulative_Market_Return'], label='Cumulative Market Return', color='orange', alpha=0.75)

plt.title('Cumulative Returns of RSI Strategy vs Market')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Comprehensive report for RSI strategy
print("\nComprehensive Report for RSI Strategy:")
print("Entry and Exit Points:")
print(rsi_signals.loc[rsi_signals['Position'] == 1, ['Close', 'RSI']])
print(rsi_signals.loc[rsi_signals['Position'] == -1, ['Close', 'RSI']])
total_percentage_increase_rsi = rsi_signals['Cumulative_Strategy_Return'].iloc[-1] * 100
print(f"Total Percentage Increase by RSI Strategy: {total_percentage_increase_rsi:.2f}%")

# Performance Metrics
annualized_return = ((1 + rsi_signals['Cumulative_Strategy_Return'].iloc[-1]) ** (365.0 / len(rsi_signals)) - 1)
annualized_volatility = rsi_signals['Strategy_Return'].std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.03) / annualized_volatility  # Assuming risk-free rate of 3%

print(f"Annualized Return: {annualized_return:.2f}")
print(f"Annualized Volatility: {annualized_volatility:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
