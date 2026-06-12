import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd

# 1. Fetch a single asset to test our execution loop safely
TICKER = "AAPL"
df = yf.download(TICKER, start="2023-01-01", end="2026-01-01")

# 2. Calculate the baseline indicators we optimized earlier
df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

df['Price_Change'] = df['Close'].diff()
df['Gain'] = np.where(df['Price_Change'] > 0, df['Price_Change'], 0)
df['Loss'] = np.where(df['Price_Change'] < 0, -df['Price_Change'], 0)
roll_gain = df['Gain'].rolling(window=14).mean()
roll_loss = df['Loss'].rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + (roll_gain / roll_loss)))
df.dropna(inplace=True)
