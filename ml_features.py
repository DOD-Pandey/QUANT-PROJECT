import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

print("--- Phase 5: Calibrated Tournament Pipeline Active ---")

# 1. Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ==============================================================================
# SPECIAL FIX: EXPLICITLY DROP RAW PRICE COLUMNS TO PREVENT BLINDING THE MODEL
# ==============================================================================
# We drop 'id', 'Date', and raw price/volume boundaries. We ONLY keep pre-computed 'feature_...' columns.
feature_cols = [col for col in train_df.columns if col.startswith('feature_')]

X_train_raw = train_df[feature_cols]
y_train = train_df['target']
X_test_raw = test_df[feature_cols]

print(f"✅ Training rows: {X_train_raw.shape[0]} | Testing rows: {X_test_raw.shape[0]}")
print(f"📋 STAGE 1 CLEANED FEATURES USED: {feature_cols}")

# 3. Time-Series Train/Validation split (80/20 chronological split)
val_split = int(len(train_df) * 0.80)
X_tr_raw, X_val_raw = X_train_raw.iloc[:val_split], X_train_raw.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

# 4. Apply Feature Scaling to Clean Stationary Indicators
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- Model 1: Regularized Ridge Baseline ---
ridge_model = Ridge(alpha=100.0) # High alpha to heavily smooth out remaining return noise
ridge_model.fit(X_tr_scaled, y_tr)
ridge_val_preds = ridge_model.predict(X_val_scaled)
print(f"\n[Baseline] Cleaned Ridge Regression Validation R² Score: {r2_score(y_val, ridge_val_preds):.6f}")

# --- Model 2: Regularized Tuned Gradient Boosting ---
gb_model = HistGradientBoostingRegressor(
    max_iter=80,            # Moderate trees to prevent over-growing into variance
    max_leaf_nodes=5,       # Shallow leaves block noise memorization
    learning_rate=0.01,     # Small learning steps maximize generalization
    l2_regularization=20.0, # High anchor penalty to damp market noise oscillations
    min_samples_leaf=30,    # Requires deep historical patterns for split confirmation
    random_state=42
)
gb_model.fit(X_tr_raw, y_tr)
gb_val_preds = gb_model.predict(X_val_raw)
print(f"[Production] Cleaned Tuned Gradient Boosting Validation R²:   {r2_score(y_val, gb_val_preds):.6f}")

# ==============================================================================
# EXPORT COMPLIANT LEADERBOARD SUBMISSION
# ==============================================================================
final_production_model = HistGradientBoostingRegressor(
    max_iter=80, max_leaf_nodes=5, learning_rate=0.01, l2_regularization=20.0, min_samples_leaf=30, random_state=42
)
final_production_model.fit(X_train_raw, y_train)
final_predictions = final_production_model.predict(X_test_raw)

submission = pd.DataFrame({
    'Row_ID': test_df.index,
    'predicted_return': final_predictions
})
submission.to_csv("submission.csv", index=False)
print("\n🚀 Submission matrices compiled successfully! file saved as 'submission.csv'.")