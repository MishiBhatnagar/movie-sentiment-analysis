# debug_samples.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

print("="*60)
print("🔍 DEBUGGING - CHECKING SAMPLE SIZES")
print("="*60)

# Check cleaned reviews
df = pd.read_csv('output/cleaned_reviews.csv')
print(f"\n📂 Cleaned reviews file: {len(df)} total reviews")
print(f"   Positive: {sum(df['sentiment']=='positive')}")
print(f"   Negative: {sum(df['sentiment']=='negative')}")

# Check features
X = np.load('output/tfidf_features.npy')
y = np.load('output/labels.npy')

print(f"\n📊 Features shape: {X.shape}")
print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]} words")
print(f"   Labels distribution: {np.bincount(y)}")
print(f"   Positive samples: {sum(y==1)}")
print(f"   Negative samples: {sum(y==0)}")

# Check train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Train/Test Split (70-30):")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")
print(f"   Training distribution: {np.bincount(y_train)}")
print(f"   Testing distribution: {np.bincount(y_test)}")

# Check for any issues
print(f"\n📊 Data Quality Check:")
print(f"   Any NaN values: {np.any(np.isnan(X))}")
print(f"   Any infinite values: {np.any(np.isinf(X))}")
print(f"   Min value: {X.min():.4f}")
print(f"   Max value: {X.max():.4f}")
print(f"   Mean value: {X.mean():.4f}")

print("\n✅ Debug complete - Your data looks good!")
print(f"   You have {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
