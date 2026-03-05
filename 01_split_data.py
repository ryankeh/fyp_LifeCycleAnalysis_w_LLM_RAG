"""
Script 1: Split the combined dataset into train, validation, and test sets
Saves: train.csv, val.csv, test.csv in the data folder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
Path("../data").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DATA SPLITTING SCRIPT")
print("=" * 60)

# Load the combined dataset
data_path = "combined_carbon_intensity_dataset.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Target variable: carbon_intensity")

# Define features (excluding non-numeric columns)
feature_cols = [col for col in df.columns if col not in 
                ['Country', 'Country Code', 'industry_code', 'industry_name', 
                 'carbon_intensity']]

print(f"Features: {len(feature_cols)} variables")

# Display basic statistics
print("\n" + "-" * 40)
print("TARGET VARIABLE STATISTICS:")
print("-" * 40)
print(f"Mean: {df['carbon_intensity'].mean():.4f}")
print(f"Std: {df['carbon_intensity'].std():.4f}")
print(f"Min: {df['carbon_intensity'].min():.4f}")
print(f"Max: {df['carbon_intensity'].max():.4f}")

# ============================================================================
# HANDLE MISSING VALUES - FIXED VERSION
# ============================================================================

print("\n" + "-" * 40)
print("HANDLING MISSING VALUES")
print("-" * 40)

# Check for missing values
print("Checking for missing values...")
missing_counts = df[feature_cols].isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

if len(missing_cols) > 0:
    print(f"\nFound {len(missing_cols)} columns with missing values:")
    for col in missing_cols.index:
        print(f"  {col}: {missing_counts[col]} missing values ({missing_counts[col]/len(df)*100:.2f}%)")
    
    # Fill missing values with column mean - FIXED VERSION
    print("\nFilling missing values with column means...")
    for col in missing_cols.index:
        col_mean = df[col].mean(skipna=True)  # Skip NaN when calculating mean
        print(f"  {col}: mean = {col_mean:.4f}")
        # Use proper pandas assignment instead of inplace fillna
        df[col] = df[col].fillna(col_mean)
    
    # Verify no missing values remain
    print("\nVerifying no missing values remain:")
    remaining_missing = df[feature_cols].isnull().sum().sum()
    print(f"  Total missing values after filling: {remaining_missing}")
    
    # Double-check with numpy
    print("\nVerifying with numpy conversion:")
    X_check = df[feature_cols].values
    print(f"  NumPy NaN count: {np.isnan(X_check).sum()}")
    print(f"  NumPy Inf count: {np.isinf(X_check).sum()}")
    
    if np.isnan(X_check).sum() > 0:
        print("\n⚠️ Still have NaN values! Using more aggressive approach...")
        # Last resort: fill with 0 for any remaining NaNs
        df[feature_cols] = df[feature_cols].fillna(0)
        print("  Filled remaining NaNs with 0")
        
        # Final check
        X_check = df[feature_cols].values
        print(f"  Final NaN count: {np.isnan(X_check).sum()}")
else:
    print("No missing values found!")

# ============================================================================
# CHECK FOR INFINITE VALUES
# ============================================================================

print("\n" + "-" * 40)
print("CHECKING FOR INFINITE VALUES")
print("-" * 40)

X_check = df[feature_cols].values
inf_count = np.isinf(X_check).sum()
print(f"Infinite values found: {inf_count}")

if inf_count > 0:
    print("Replacing infinite values with column means...")
    # Replace inf with NaN first, then fill with mean
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    for col in feature_cols:
        col_mean = df[col].mean(skipna=True)
        if np.isnan(col_mean):  # If mean is NaN (all values were inf)
            col_mean = 0
        df[col] = df[col].fillna(col_mean)
    
    # Final check
    X_check = df[feature_cols].values
    print(f"  Final infinite count: {np.isinf(X_check).sum()}")

# ============================================================================
# DATA TYPES
# ============================================================================

print("\n" + "-" * 40)
print("DATA TYPES:")
print("-" * 40)
print(df[feature_cols].dtypes.value_counts())

# ============================================================================
# SPLITTING DATA
# ============================================================================

print("\n" + "-" * 40)
print("SPLITTING DATA")
print("-" * 40)

# Define features and target
target_col = 'carbon_intensity'
X = df[feature_cols]
y = df[target_col]

print(f"Features: {len(feature_cols)} variables")
print(f"Features list: {feature_cols[:5]}... (showing first 5)")

# First split: separate test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

# Second split: separate train (80% of original) and validation (10% of original)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, shuffle=True
)  # 0.1111 of 90% = 10% of total

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train):,} rows ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Validation: {len(X_val):,} rows ({len(X_val)/len(df)*100:.1f}%)")
print(f"  Test: {len(X_test):,} rows ({len(X_test)/len(df)*100:.1f}%)")

# ============================================================================
# CREATE DATAFRAMES WITH ALL COLUMNS
# ============================================================================

# Reconstruct full dataframes with all columns
train_df = df.loc[X_train.index].copy()
val_df = df.loc[X_val.index].copy()
test_df = df.loc[X_test.index].copy()

# Verify no NaN in splits
print("\n" + "-" * 40)
print("VERIFYING SPLITS:")
print("-" * 40)

for name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    X_split = split_df[feature_cols].values
    print(f"{name} set - NaN: {np.isnan(X_split).sum()}, Inf: {np.isinf(X_split).sum()}")

# Verify target statistics in each split
print("\n" + "-" * 40)
print("TARGET STATISTICS BY SPLIT:")
print("-" * 40)

for name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    print(f"\n{name} Set:")
    print(f"  Mean: {split_df['carbon_intensity'].mean():.4f}")
    print(f"  Std: {split_df['carbon_intensity'].std():.4f}")
    print(f"  Min: {split_df['carbon_intensity'].min():.4f}")
    print(f"  Max: {split_df['carbon_intensity'].max():.4f}")

# ============================================================================
# SAVE SPLITS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING SPLITS")
print("-" * 40)

train_path = "../data/train.csv"
val_path = "../data/val.csv"
test_path = "../data/test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Train set saved to: {train_path}")
print(f"Validation set saved to: {val_path}")
print(f"Test set saved to: {test_path}")

# Save feature names and info for later use
feature_info = {
    'feature_names': feature_cols,
    'target_name': target_col,
    'feature_count': len(feature_cols),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df)
}

with open('../data/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("\nFeature information saved to: ../data/feature_info.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("SPLITTING COMPLETE - SUMMARY")
print("=" * 60)
print(f"Total samples: {len(df):,}")
print(f"Training samples: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation samples: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test samples: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
print(f"Features: {len(feature_cols)}")
if 'missing_cols' in locals():
    print(f"Missing values handled: {len(missing_cols)} columns")
print("=" * 60)