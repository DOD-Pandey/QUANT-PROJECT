import numpy as np
import pandas as pd

# 1. Create a dummy dataset to test our feature formulas step-by-step
np.random.seed(42)
dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
mock_close = 100 * np.exp(np.random.normal(0.001, 0.01, 100).cumsum())
mock_volume = np.random.randint(50000, 150000, 100)

df = pd.DataFrame({"Close": mock_close, "Volume": mock_volume}, index=dates)
print("Initial dataframe columns:", df.columns)

# Calculate daily log returns using numpy's log and pandas shift
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# 20-day Simple Moving Average (Smoothed trend level)
df["sma_20"] = df["Close"].rolling(20).mean()
# 20-day Rolling Volatility (Recent asset risk)
df["volatility_20"] = df["log_return"].rolling(20).std()
# Price vs SMA (How far absolute price deviates from its trend line)
df["price_vs_sma"] = (df["Close"] - df["sma_20"]) / df["sma_20"]

# 12-day Exponential Moving Average (Reacts faster than SMA)
df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()

# 14-day Relative Strength Index (RSI)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# Volume Ratio: Is today's volume unusual compared to the 20-day average?
df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
# Volume Spike Indicator (Binary: 1 if volume is double its mean, else 0)
df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

# Lag features: shifting historical returns forward
df["return_lag_1"] = df["log_return"].shift(1)
df["return_lag_5"] = df["log_return"].shift(5)

# Volatility Ratio: Short-term volatility vs. Long-term volatility
df["vol_5"] = df["log_return"].rolling(5).std()
df["vol_20"] = df["log_return"].rolling(20).std()
df["vol_ratio"] = df["vol_5"] / df["vol_20"]

# Preview the engineered features matrix
print("\nEngineered Feature Columns Matrix:")
print(df.columns.tolist())

# ==============================================================================
# BLOCK 3 & 4: MISSING VALUES, LEAKAGE CHECK, AND CORRELATION
# ==============================================================================

# 1. Clean missing values caused by rolling windows safely (Safest approach for the project)
df_clean = df.dropna().copy()
print(f"\nRemaining data rows after dropping NaN windows: {len(df_clean)}")
# 2. Define the list of features we want to check
feature_cols = ['sma_20', 'volatility_20', 'price_vs_sma', 'rsi_14', 
                'volume_ratio', 'return_lag_1', 'return_lag_5', 'vol_ratio']

# 3. Compute the pairwise correlation matrix to check for redundant features
corr_matrix = df_clean[feature_cols].corr()
# 4. Print the raw correlation matrix to your terminal
print("\nPairwise Feature Correlation Matrix:")
print(corr_matrix.round(2))

import matplotlib.pyplot as plt
import seaborn as sns

# 5. Set up the matplotlib figure canvas
plt.figure(figsize=(10, 8))
# 6. Generate a divergence heatmap (Red = Positive, Blue = Negative, White = Neutral)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
)
# 7. Add titles and adjust layout so nothing gets clipped
plt.title("Feature Correlation Matrix Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
# 8. Render the plot to your screen
plt.show()
