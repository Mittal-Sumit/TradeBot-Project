import pandas as pd
import numpy as np

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
})

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

# Display the RSI signals DataFrame
print("\nRSI Signals DataFrame:")
print(rsi_signals)

# Comprehensive report for RSI strategy
print("\nComprehensive Report for RSI Strategy:")
print("Entry and Exit Points:")
print(rsi_signals.loc[rsi_signals['Position'] == 1, ['Close', 'RSI']])
print(rsi_signals.loc[rsi_signals['Position'] == -1, ['Close', 'RSI']])
total_percentage_increase_rsi = rsi_signals['Cumulative_Strategy_Return'].iloc[-1] * 100
print(f"Total Percentage Increase by RSI Strategy: {total_percentage_increase_rsi:.2f}%")