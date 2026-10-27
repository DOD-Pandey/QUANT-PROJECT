import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd

# 1. Define Dataset Environment
TICKERS = ["AAPL", "GLD", "TLT"]
df = yf.download(TICKERS, start="2023-01-01", end="2026-01-01")

# 2. Compute Multi-Asset Indicators
for ticker in TICKERS:
    df[('EMA', ticker)] = df[('Close', ticker)].ewm(span=14, adjust=False).mean()
    df[('Price_Change', ticker)] = df[('Close', ticker)].diff()
    df[('Gain', ticker)] = np.where(df[('Price_Change', ticker)] > 0, df[('Price_Change', ticker)], 0)
    df[('Loss', ticker)] = np.where(df[('Price_Change', ticker)] < 0, -df[('Price_Change', ticker)], 0)
    
    roll_gain = df[('Gain', ticker)].rolling(window=14).mean()
    roll_loss = df[('Loss', ticker)].rolling(window=14).mean()
    df[('RSI', ticker)] = 100 - (100 / (1 + (roll_gain / roll_loss)))

# 3. Dynamic Strategy Signal Allocation Matrix
for ticker in TICKERS:
    df[('Signal', ticker)] = 0
    buy_cond = (df[('Close', ticker)] > df[('EMA', ticker)]) & (df[('RSI', ticker)] < 80)
    df.loc[buy_cond, ('Signal', ticker)] = 1
    
    exit_cond = (df[('Close', ticker)] < df[('EMA', ticker)]) | (df[('RSI', ticker)] > 80)
    df.loc[exit_cond, ('Signal', ticker)] = 0

df['Total_Active_Signals'] = df['Signal'].sum(axis=1)

for ticker in TICKERS:
    df[('Weight', ticker)] = np.where(
        df['Total_Active_Signals'] > 0,
        df[('Signal', ticker)] / df['Total_Active_Signals'],
        0
    )

# 4. Run Backtest Simulation Engine
df['Portfolio_Daily_Return'] = 0.0

for ticker in TICKERS:
    df[('Shifted_Weight', ticker)] = df[('Weight', ticker)].shift(1)
    df[('Log_Return', ticker)] = np.log(df[('Close', ticker)] / df[('Close', ticker)].shift(1))
    df['Portfolio_Daily_Return'] += df[('Shifted_Weight', ticker)] * df[('Log_Return', ticker)]

df.dropna(inplace=True)

# 5. Extract Final Performance Values
port_annualized_return = df['Portfolio_Daily_Return'].mean() * 252
port_annualized_vol = df['Portfolio_Daily_Return'].std() * np.sqrt(252)
port_sharpe = port_annualized_return / port_annualized_vol

# 6. Display Production Terminal Report
print("\n" + "="*50)
print("     FINAL MULTI-ASSET PORTFOLIO RESULTS      ")
print("="*50)
print(f"Annualized Return:     {port_annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {port_annualized_vol * 100:.2f}%")
print(f"Annualized Sharpe:     {port_sharpe:.2f}")
print("====================================" + "\n")
