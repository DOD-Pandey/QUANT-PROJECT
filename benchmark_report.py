import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define ticker as a variable at the top (Checklist item!)
TICKER = "SPY"

# Fetch 3 years of daily historical data
df = yf.download(TICKER, start="2023-01-01", end="2026-01-01")

# Clean data: Display data structure to confirm download
#print(df.head())
# 1. Calculate Daily Log Returns using the correct column structure
df['Log_Return'] = np.log(df[('Close', 'SPY')] / df[('Close', 'SPY')].shift(1))
df.dropna(inplace=True)

# 2. Compute Benchmark Metrics
mean_daily_return = df['Log_Return'].mean()
annualized_return = mean_daily_return * 252

daily_volatility = df['Log_Return'].std()
annualized_volatility = daily_volatility * np.sqrt(252)

# 3. Sharpe Ratio (Assuming risk-free rate = 0 per instructions)
annualized_sharpe = annualized_return / annualized_volatility

# 4. Calculate Maximum Drawdown from the Cumulative Equity Curve
df['Cumulative_Log_Returns'] = df['Log_Return'].cumsum()
df['Equity_Curve'] = np.exp(df['Cumulative_Log_Returns'])
df['Peak'] = df['Equity_Curve'].cummax()
df['Drawdown'] = (df['Equity_Curve'] - df['Peak']) / df['Peak']
max_drawdown = df['Drawdown'].min()

# 5. Best and worst single days
best_day = df['Log_Return'].max()
worst_day = df['Log_Return'].min()

# 6. Print out the formal Benchmark Report
print("\n" + "="*40)
print(f" BUY-AND-HOLD BENCHMARK REPORT FOR {TICKER} ")
print("="*40)
print(f"Annualized Return:     {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
print(f"Annualized Sharpe:     {annualized_sharpe:.2f}")
print(f"Maximum Drawdown:      {max_drawdown * 100:.2f}%")
print(f"Best Single Day:       {best_day * 100:.2f}%")
print(f"Worst Single Day:      {worst_day * 100:.2f}%")
print("="*40 + "\n")

import scipy.stats as stats

# 1. Create a figure container
plt.figure(figsize=(10, 6))

# 2. Plot the actual data histogram
count, bins, ignored = plt.hist(df['Log_Return'], bins=50, density=True, 
                                alpha=0.6, color='b', edgecolor='black', 
                                label='Actual Log Returns')

# 3. Mathematically model a perfect theoretical normal curve overlay
mu = df['Log_Return'].mean()
sigma = df['Log_Return'].std()
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')

# 4. Add chart details and styling
plt.title(f'{TICKER} Log Returns Distribution vs. Normal Curve')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. CRITICAL: Save the file directly to your project folder
plt.savefig('benchmark_histogram.png', dpi=300)
print("Graph successfully generated and saved as 'benchmark_histogram.png'!")
