"""
Script B3: Prepare inputs for meta-learner
Combines predictions from industry BNN, country BNN, and kriging models
Uses Country Code and industry_code as composite keys for merging
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import r2_score

# Create directories
Path("../results/bayesian/meta").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PREPARING META-LEARNER INPUTS")
print("=" * 60)

# ============================================================================
# LOAD ORIGINAL DATA
# ============================================================================

print("\n" + "-" * 40)
print("LOADING ORIGINAL DATA")
print("-" * 40)

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

# Add split identifier
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Combine all data
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total samples: {len(all_df)}")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# ============================================================================
# LOAD ALL PREDICTIONS
# ============================================================================

print("\n" + "-" * 40)
print("LOADING PREDICTIONS")
print("-" * 40)

# Industry BNN predictions (all splits)
industry_train = pd.read_csv("../results/bayesian/industry_predictions_train.csv")
industry_val = pd.read_csv("../results/bayesian/industry_predictions_val.csv")
industry_test = pd.read_csv("../results/bayesian/industry_predictions_test.csv")
industry_df = pd.concat([industry_train, industry_val, industry_test], ignore_index=True)
print(f"Industry predictions: {len(industry_df)} samples")

# Country BNN predictions (all splits)
country_train = pd.read_csv("../results/bayesian/country_predictions_train.csv")
country_val = pd.read_csv("../results/bayesian/country_predictions_val.csv")
country_test = pd.read_csv("../results/bayesian/country_predictions_test.csv")
country_df = pd.concat([country_train, country_val, country_test], ignore_index=True)
print(f"Country predictions: {len(country_df)} samples")

# Kriging predictions (all splits) - these already exist from your previous run
kriging_train = pd.read_csv("../results/bayesian/kriging_train_predictions.csv")
kriging_val = pd.read_csv("../results/bayesian/kriging_val_predictions.csv")
kriging_test = pd.read_csv("../results/bayesian/kriging_test_predictions.csv")
kriging_df = pd.concat([kriging_train, kriging_val, kriging_test], ignore_index=True)
print(f"Kriging predictions: {len(kriging_df)} samples")

# ============================================================================
# VERIFY COMPOSITE KEYS
# ============================================================================

print("\n" + "-" * 40)
print("VERIFYING COMPOSITE KEYS")
print("-" * 40)

# Check for duplicates in each prediction set
for name, df in [('Industry', industry_df), ('Country', country_df), ('Kriging', kriging_df)]:
    duplicates = df.duplicated(subset=['Country Code', 'industry_code']).sum()
    print(f"{name}: {duplicates} duplicate (Country Code, industry_code) pairs")
    
    # Check if all required columns exist
    if name == 'Kriging':
        required_cols = ['Country Code', 'industry_code', 'predicted', 'std']
    else:
        required_cols = ['Country Code', 'industry_code', f'{name.lower()}_pred', f'{name.lower()}_std']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")

# ============================================================================
# MERGE ALL PREDICTIONS USING COMPOSITE KEYS
# ============================================================================

print("\n" + "-" * 40)
print("MERGING PREDICTIONS")
print("-" * 40)

# Start with original data
meta_df = all_df[['Country Code', 'industry_code', 'split', 'Country', 
                  'industry_name', 'carbon_intensity']].copy()

# Merge industry predictions
meta_df = meta_df.merge(
    industry_df[['Country Code', 'industry_code', 'industry_pred', 'industry_std']],
    on=['Country Code', 'industry_code'],
    how='left'
)
print(f"After industry merge: {meta_df['industry_pred'].notna().sum()} non-null")

# Merge country predictions
meta_df = meta_df.merge(
    country_df[['Country Code', 'industry_code', 'country_pred', 'country_std']],
    on=['Country Code', 'industry_code'],
    how='left'
)
print(f"After country merge: {meta_df['country_pred'].notna().sum()} non-null")

# Merge kriging predictions
meta_df = meta_df.merge(
    kriging_df[['Country Code', 'industry_code', 'predicted', 'std']],
    on=['Country Code', 'industry_code'],
    how='left'
).rename(columns={'predicted': 'krige_pred', 'std': 'krige_std'})
print(f"After kriging merge: {meta_df['krige_pred'].notna().sum()} non-null")

# Check for missing values
print("\nMissing values per column:")
print(meta_df.isnull().sum())

# Verify we have the right number of samples
print(f"\nFinal samples: {len(meta_df)} (should be {len(all_df)})")

# ============================================================================
# CREATE FEATURE MATRIX
# ============================================================================

print("\n" + "-" * 40)
print("CREATING META-LEARNER FEATURES")
print("-" * 40)

feature_cols = ['industry_pred', 'industry_std', 'country_pred', 'country_std', 
                'krige_pred', 'krige_std']
target_col = 'carbon_intensity'

# Drop any rows with missing predictions
initial_len = len(meta_df)
meta_df = meta_df.dropna(subset=feature_cols)
print(f"Dropped {initial_len - len(meta_df)} rows with missing predictions")
print(f"Final samples: {len(meta_df)}")

# Create feature matrix
X_meta = meta_df[feature_cols].values.astype(np.float32)
y_meta = meta_df[target_col].values.astype(np.float32)
splits = meta_df['split'].values

print(f"\nFinal samples by split:")
for split in ['train', 'val', 'test']:
    mask = splits == split
    print(f"  {split}: {mask.sum()} samples")

# ============================================================================
# QUICK PERFORMANCE CHECK
# ============================================================================

print("\n" + "-" * 40)
print("PERFORMANCE CHECK")
print("-" * 40)

for split in ['train', 'val', 'test']:
    mask = splits == split
    if mask.sum() == 0:
        continue
    
    print(f"\n{split.upper()} SET ({mask.sum()} samples):")
    print(f"  Industry BNN R²: {r2_score(y_meta[mask], X_meta[mask, 0]):.4f}")
    print(f"  Country BNN R²: {r2_score(y_meta[mask], X_meta[mask, 2]):.4f}")
    print(f"  Kriging R²: {r2_score(y_meta[mask], X_meta[mask, 4]):.4f}")

# ============================================================================
# SAVE DATA
# ============================================================================

print("\n" + "-" * 40)
print("SAVING META-LEARNER DATA")
print("-" * 40)

# Split into train/val/test
train_mask = splits == 'train'
val_mask = splits == 'val'
test_mask = splits == 'test'

X_train = X_meta[train_mask]
y_train = y_meta[train_mask]
X_val = X_meta[val_mask]
y_val = y_meta[val_mask]
X_test = X_meta[test_mask]
y_test = y_meta[test_mask]

# Save features and targets
np.save('../results/bayesian/meta/X_train.npy', X_train)
np.save('../results/bayesian/meta/y_train.npy', y_train)
np.save('../results/bayesian/meta/X_val.npy', X_val)
np.save('../results/bayesian/meta/y_val.npy', y_val)
np.save('../results/bayesian/meta/X_test.npy', X_test)
np.save('../results/bayesian/meta/y_test.npy', y_test)

# Save feature names
with open('../results/bayesian/meta/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

# Save metadata for reference
meta_df[['Country Code', 'industry_code', 'split', 'carbon_intensity'] + feature_cols].to_csv(
    '../results/bayesian/meta/meta_data.csv', index=False)

print(f"\nSaved:")
print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")
print("=" * 60)