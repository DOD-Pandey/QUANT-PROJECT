import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the datasets
print("Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Basic shape and info
print(f"Train set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# 3. Check for missing values
missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("\n--- Missing Values in Train Set ---")
print(missing_values)

# 4. Target distribution statistics
print("\n--- Target Statistics ---")
print(train['target'].describe())

# 5. Plot Target Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['target'], bins=100, kde=True, color='blue')
plt.title('Target Distribution')
plt.xlabel('Forward Return')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# 6. Quick look at data types and first few rows
print("\n--- First 3 Rows ---")
print(train.head(3))
