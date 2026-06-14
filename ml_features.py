import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score

print("--- Phase 4: Leaderboard Pipeline Active ---")

# 1. Load the official competition files directly
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"✅ Loaded Training Data: {train_df.shape[0]} rows")
print(f"✅ Loaded Hidden Test Data: {test_df.shape[0]} rows")

# 2. Identify available feature columns automatically (ignoring date/target)
ignore_cols = ['date', 'Date', 'target', 'Target']
feature_cols = [col for col in train_df.columns if col not in ignore_cols]

# 3. Create clean feature and target matrices
X_train = train_df[feature_cols]
y_train = train_df['target']

# Note: test.csv doesn't have a known target column because it's a hidden test set!
X_test = test_df[feature_cols]

print(f"Features being used for training: {feature_cols}")

# 4. Train your absolute best performing model (The Gradient Booster!)
final_model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
final_model.fit(X_train, y_train)
print("✨ Final Model successfully trained on 100% of train.csv data.")

# 5. Generate your tournament submission predictions!
test_predictions = final_model.predict(X_test)

# 6. Format your submission file exactly how the leaderboard wants it
submission = pd.DataFrame({
    'Row_ID': test_df.index,       # Alternately use test_df['date'] if requested
    'predicted_return': test_predictions
})

submission.to_csv("submission.csv", index=False)
print("🚀 Victory! Your leaderboard predictions are saved cleanly as 'submission.csv'.")
