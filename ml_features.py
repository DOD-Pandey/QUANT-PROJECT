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
# plt.show()  # Commented out so the script runs straight through to the models!

# The target: Tomorrow's return, aligned to today's feature row
df['target'] = df['log_return'].shift(-1)

# ==============================================================================
# PHASE 3: ML MODELING ENGINE (BLOCK 1 & 2)
# ==============================================================================

# 1. Create the prediction target (Tomorrow's return)
df['target'] = df['log_return'].shift(-1)

# Drop any remaining NaNs safely after target creation
df_ml = df.dropna().copy()

# 2. Define features (X) and target (y) Matrices
feature_cols = ['sma_20', 'volatility_20', 'price_vs_sma', 'rsi_14', 
                'volume_ratio', 'return_lag_1', 'return_lag_5', 'vol_ratio']

X = df_ml[feature_cols]
y = df_ml['target']

# 3. Chronological Time-Series Split (80% Train, 20% Test) - NO SHUFFLING!
split_idx = int(len(df_ml) * 0.80)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("\n--- Block 1: Data Dimensions ---")
print(f"Training set rows: {X_train.shape[0]} | Testing set rows: {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

print("\n--- Block 2: Training Baselines ---")

# 4. Model 1: Linear Regression Baseline
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)

print(f"Linear Regression Baseline Test R² Score: {lr_r2:.6f}")
print("Linear Regression Coefficients (Feature Weights):")
for name, coef in zip(feature_cols, lr_model.coef_):
    print(f"  {name}: {coef:.6f}")

# 5. Model 2: Decision Tree Baseline (Shallow depth to restrict overfitting)
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_preds)

print(f"\nDecision Tree Test R² Score: {dt_r2:.6f}")
# ==============================================================================
# BLOCK 3: RANDOM FOREST WORKHORSE (XGBoost bypassed)
# ==============================================================================
from sklearn.ensemble import RandomForestRegressor

print("\n--- Block 3: Training Ensemble Workhorses ---")

# 6. Model 3: Random Forest (Averaging trees to squash variance noise)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)

print(f"Random Forest Test R² Score: {rf_r2:.6f}")

# (XGBoost is commented out for now to bypass the Anaconda installation block)
# from xgboost import XGBRegressor
# xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
# xgb_model.fit(X_train, y_train)
# xgb_preds = xgb_model.predict(X_test)
# xgb_r2 = r2_score(y_test, xgb_preds)
# print(f"XGBoost Test R² Score: {xgb_r2:.6f}")


# ==============================================================================
# BLOCK 4: FEATURE IMPORTANCE DIAGNOSTICS
# ==============================================================================
import matplotlib.pyplot as plt

print("\n--- Block 4: Plotting Feature Importances ---")

# Extract importance values from our trained Random Forest
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)

# Generate a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], color="skyblue", edgecolor="black")
plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
plt.xlabel('Relative Importance Score', fontsize=12, fontweight='bold')
plt.title('What Did Our Model Learn? (Random Forest Feature Importance)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot directly as an image file so it doesn't freeze your terminal!
plt.savefig('feature_importance.png')
print("🚀 Success! Feature importance plot has been saved cleanly as 'feature_importance.png'.")


