"""
Script 9: Support Vector Regression for Carbon Intensity Prediction
Uses: train.csv, val.csv, test.csv
Saves: svr_results.json, svr_predictions.csv, visualizations
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/plots").mkdir(parents=True, exist_ok=True)
Path("../models").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SUPPORT VECTOR REGRESSION SCRIPT")
print("=" * 60)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA")
print("-" * 40)

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

print(f"Train set: {len(train_df):,} rows")
print(f"Validation set: {len(val_df):,} rows")
print(f"Test set: {len(test_df):,} rows")

# Load feature info
with open('../data/feature_info.json', 'r') as f:
    feature_info = json.load(f)

feature_cols = feature_info['feature_names']
target_col = feature_info['target_name']

print(f"\nFeatures ({len(feature_cols)}):")
for i, col in enumerate(feature_cols[:10]):
    print(f"  {i+1}. {col}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")

# Prepare data
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df[target_col].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df[target_col].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df[target_col].values.astype(np.float32)

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================================
# VERIFY NO NAN VALUES
# ============================================================================

print("\n" + "-" * 40)
print("VERIFYING NO NAN VALUES")
print("-" * 40)

for name, X in [('Train', X_train), ('Validation', X_val), ('Test', X_test)]:
    print(f"{name} set NaN count: {np.isnan(X).sum()}")
    print(f"{name} set Inf count: {np.isinf(X).sum()}")

# ============================================================================
# SCALING IS CRITICAL FOR SVR
# ============================================================================

print("\n" + "-" * 40)
print("SCALING FEATURES (CRITICAL FOR SVR)")
print("-" * 40)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scale target as well (helps with SVR convergence)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Save scalers
joblib.dump(scaler_X, '../models/svr_scaler_X.pkl')
joblib.dump(scaler_y, '../models/svr_scaler_y.pkl')
print("Scalers saved to: ../models/")

print(f"\nAfter scaling - X_train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"After scaling - y_train mean: {y_train_scaled.mean():.6f}, std: {y_train_scaled.std():.6f}")

# ============================================================================
# DEFINE METRICS FUNCTIONS
# ============================================================================

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zeros by adding small epsilon"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Calculate all metrics"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'MedAE': np.median(np.abs(y_true - y_pred)),
        'Max_Error': np.max(np.abs(y_true - y_pred))
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  SMAPE: {metrics['SMAPE']:.2f}%")
    
    return metrics

# ============================================================================
# CREATE SUBSET FOR FASTER TUNING
# ============================================================================

print("\n" + "-" * 40)
print("CREATING SUBSET FOR FASTER TUNING")
print("-" * 40)

# Use 10,000 samples for tuning (good balance of speed and representativeness)
TUNE_SAMPLE_SIZE = 10000
np.random.seed(42)  # For reproducibility

# Sample from training data
tune_indices = np.random.choice(len(X_train_scaled), TUNE_SAMPLE_SIZE, replace=False)
X_tune = X_train_scaled[tune_indices]
y_tune = y_train_scaled[tune_indices]

# Also create a smaller validation set for tuning (use validation samples)
VAL_TUNE_SIZE = min(3000, len(X_val_scaled))
val_indices = np.random.choice(len(X_val_scaled), VAL_TUNE_SIZE, replace=False)
X_val_tune = X_val_scaled[val_indices]
y_val_tune = y_val_scaled[val_indices]
y_val_orig_tune = y_val[val_indices]

print(f"Tuning subset: {TUNE_SAMPLE_SIZE:,} samples ({TUNE_SAMPLE_SIZE/len(X_train_scaled)*100:.1f}% of training data)")
print(f"Validation subset: {VAL_TUNE_SIZE:,} samples")
print(f"Expected speedup: ~{(len(X_train_scaled)/TUNE_SAMPLE_SIZE)**2:.0f}x faster!")

# ============================================================================
# SEQUENTIAL HYPERPARAMETER TUNING WITH VALIDATION SET (USING SUBSET)
# ============================================================================

print("\n" + "-" * 40)
print("SEQUENTIAL HYPERPARAMETER TUNING (ON SUBSET)")
print("-" * 40)
print("Using smart sequential search on 10k samples...")

best_val_rmse = float('inf')
best_params = {}
best_model_tune = None  # Model trained on subset (just for tracking)
results = []
tuning_history = []

start_time = time.time()

# ============================================================================
# STAGE 1: Find best kernel (3 runs)
# ============================================================================
print("\n" + "=" * 40)
print("STAGE 1: Finding Best Kernel (on subset)")
print("=" * 40)

stage1_params = [
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1, 'degree': 3},
    {'kernel': 'sigmoid', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1}
]

kernel_scores = []

for params in stage1_params:
    print(f"\n▶ Testing: {params['kernel']} kernel")
    
    model = SVR(**params, cache_size=1000, verbose=False)
    model.fit(X_tune, y_tune)  # Using subset!
    
    # Predict on validation subset
    y_pred_val_scaled = model.predict(X_val_tune)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
    
    val_rmse = rmse(y_val_orig_tune, y_pred_val)
    kernel_scores.append((params['kernel'], val_rmse))
    
    # Store results
    result_entry = {
        'stage': 1,
        'kernel': params['kernel'],
        'C': params['C'],
        'gamma': str(params['gamma']),
        'epsilon': params['epsilon'],
        'val_rmse': val_rmse
    }
    if 'degree' in params:
        result_entry['degree'] = params['degree']
    results.append(result_entry)
    tuning_history.append(result_entry)
    
    print(f"  Validation RMSE: {val_rmse:.6f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params.copy()
        best_model_tune = model
        print(f"  ✓ New best model!")

# Select best kernel
best_kernel = min(kernel_scores, key=lambda x: x[1])[0]
best_kernel_rmse = min(kernel_scores, key=lambda x: x[1])[1]
print(f"\n✓ Best kernel: {best_kernel} (RMSE: {best_kernel_rmse:.6f})")

# ============================================================================
# STAGE 2: Tune C parameter with best kernel (4 runs)
# ============================================================================
print("\n" + "=" * 40)
print(f"STAGE 2: Tuning C parameter with {best_kernel} kernel (on subset)")
print("=" * 40)

# Log-spaced C values
C_values = [0.1, 1.0, 10.0]
C_scores = []

for C in C_values:
    print(f"\n▶ Testing C = {C}")
    
    params = {
        'kernel': best_kernel,
        'C': C,
        'gamma': 'scale',
        'epsilon': 0.1
    }
    if best_kernel == 'poly':
        params['degree'] = 3
    
    model = SVR(**params, cache_size=1000, verbose=False)
    model.fit(X_tune, y_tune)  # Using subset!
    
    y_pred_val_scaled = model.predict(X_val_tune)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
    
    val_rmse = rmse(y_val_orig_tune, y_pred_val)
    C_scores.append((C, val_rmse))
    
    # Store results
    result_entry = {
        'stage': 2,
        'kernel': best_kernel,
        'C': C,
        'gamma': 'scale',
        'epsilon': 0.1,
        'val_rmse': val_rmse
    }
    results.append(result_entry)
    tuning_history.append(result_entry)
    
    print(f"  Validation RMSE: {val_rmse:.6f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params.copy()
        best_model_tune = model
        print(f"  ✓ New best model!")

# Find best C
best_C = min(C_scores, key=lambda x: x[1])[0]
print(f"\n✓ Best C: {best_C}")

# Refine around best C if it's at the edge
refinement_runs = 0

if best_C == C_values[0]:  # If best is lowest value
    print("\nRefining: Testing lower C values...")
    lower_C = [0.01, 0.001]
    for C in lower_C:
        refinement_runs += 1
        params['C'] = C
        model = SVR(**params, cache_size=1000, verbose=False)
        model.fit(X_tune, y_tune)  # Using subset!
        y_pred_val_scaled = model.predict(X_val_tune)
        y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
        val_rmse = rmse(y_val_orig_tune, y_pred_val)
        
        # Store results
        result_entry = {
            'stage': '2_refine',
            'kernel': best_kernel,
            'C': C,
            'gamma': 'scale',
            'epsilon': 0.1,
            'val_rmse': val_rmse
        }
        results.append(result_entry)
        tuning_history.append(result_entry)
        
        print(f"  C={C}: RMSE={val_rmse:.6f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = params.copy()
            best_model_tune = model
            best_C = C
            print(f"  ✓ New best model!")

# ============================================================================
# STAGE 3: Tune gamma with best kernel and C (5 runs)
# ============================================================================
print("\n" + "=" * 40)
print(f"STAGE 3: Tuning gamma parameter (on subset)")
print("=" * 40)

gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
gamma_scores = []

for gamma in gamma_values:
    print(f"\n▶ Testing gamma = {gamma}")
    
    params = {
        'kernel': best_kernel,
        'C': best_C,
        'gamma': gamma,
        'epsilon': 0.1
    }
    if best_kernel == 'poly':
        params['degree'] = 3
    
    model = SVR(**params, cache_size=1000, verbose=False)
    model.fit(X_tune, y_tune)  # Using subset!
    
    y_pred_val_scaled = model.predict(X_val_tune)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
    
    val_rmse = rmse(y_val_orig_tune, y_pred_val)
    gamma_scores.append((gamma, val_rmse))
    
    # Store results
    result_entry = {
        'stage': 3,
        'kernel': best_kernel,
        'C': best_C,
        'gamma': str(gamma),
        'epsilon': 0.1,
        'val_rmse': val_rmse
    }
    results.append(result_entry)
    tuning_history.append(result_entry)
    
    print(f"  Validation RMSE: {val_rmse:.6f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params.copy()
        best_model_tune = model
        print(f"  ✓ New best model!")

best_gamma = min(gamma_scores, key=lambda x: x[1])[0]
print(f"\n✓ Best gamma: {best_gamma}")

# ============================================================================
# STAGE 4: Tune epsilon with all best parameters (5 runs)
# ============================================================================
print("\n" + "=" * 40)
print(f"STAGE 4: Tuning epsilon parameter (on subset)")
print("=" * 40)

epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.5]
epsilon_scores = []

for epsilon in epsilon_values:
    print(f"\n▶ Testing epsilon = {epsilon}")
    
    params = {
        'kernel': best_kernel,
        'C': best_C,
        'gamma': best_gamma,
        'epsilon': epsilon
    }
    if best_kernel == 'poly':
        params['degree'] = 3
    
    model = SVR(**params, cache_size=1000, verbose=False)
    model.fit(X_tune, y_tune)  # Using subset!
    
    y_pred_val_scaled = model.predict(X_val_tune)
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
    
    val_rmse = rmse(y_val_orig_tune, y_pred_val)
    epsilon_scores.append((epsilon, val_rmse))
    
    # Store results
    result_entry = {
        'stage': 4,
        'kernel': best_kernel,
        'C': best_C,
        'gamma': str(best_gamma),
        'epsilon': epsilon,
        'val_rmse': val_rmse
    }
    results.append(result_entry)
    tuning_history.append(result_entry)
    
    print(f"  Validation RMSE: {val_rmse:.6f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params.copy()
        best_model_tune = model
        print(f"  ✓ New best model!")

best_epsilon = min(epsilon_scores, key=lambda x: x[1])[0]
print(f"\n✓ Best epsilon: {best_epsilon}")

tuning_time = time.time() - start_time

print("\n" + "=" * 60)
print("SEQUENTIAL TUNING COMPLETE (ON SUBSET)")
print("=" * 60)
print(f"Total iterations: {len(results)}")
print(f"Tuning time: {tuning_time:.2f} seconds ({tuning_time/60:.2f} minutes)")
print(f"Best parameters found on subset:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"Best validation RMSE (on subset): {best_val_rmse:.6f}")

# Save tuning results
tuning_df = pd.DataFrame(tuning_history)
tuning_df.to_csv('../results/svr_tuning_results.csv', index=False)
print("\nTuning results saved to: ../results/svr_tuning_results.csv")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS ON FULL DATA
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING FINAL MODEL ON FULL TRAINING DATA")
print("-" * 40)
print(f"Training on full dataset: {len(X_train_scaled):,} samples")
print("This may take a while but it's just one fit...")

final_start = time.time()

final_model = SVR(
    **best_params,
    cache_size=2000,  # Increase cache for full dataset
    verbose=True
)

final_model.fit(X_train_scaled, y_train_scaled)  # Full training data!

final_train_time = time.time() - final_start
print(f"Final model trained in {final_train_time:.2f} seconds ({final_train_time/60:.2f} minutes)")

# Save model
joblib.dump(final_model, '../models/svr_best_model.pkl')
print("Model saved to: ../models/svr_best_model.pkl")

# ============================================================================
# PREDICTIONS ON ALL SETS
# ============================================================================

print("\n" + "-" * 40)
print("GENERATING PREDICTIONS")
print("-" * 40)

# Predict in scaled space
y_pred_train_scaled = final_model.predict(X_train_scaled)
y_pred_val_scaled = final_model.predict(X_val_scaled)
y_pred_test_scaled = final_model.predict(X_test_scaled)

# Inverse transform to original scale
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).flatten()
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

print("Predictions generated for all sets")

# ============================================================================
# EVALUATE ON ALL SETS
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATION RESULTS (Final Model on Full Data)")
print("-" * 40)

train_metrics = evaluate_predictions(y_train, y_pred_train, "TRAIN SET")
val_metrics = evaluate_predictions(y_val, y_pred_val, "VALIDATION SET")
test_metrics = evaluate_predictions(y_test, y_pred_test, "TEST SET")

# ============================================================================
# REST OF YOUR CODE (Residual Analysis, Visualizations, etc.) - UNCHANGED
# ============================================================================

# [Keep all your existing code from here onward - it's perfect!]
# Just make sure to use the final_model and predictions you just created

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("RESIDUAL ANALYSIS")
print("-" * 40)

residuals_test = y_test - y_pred_test

print(f"\nTest Set Residual Statistics:")
print(f"  Mean: {np.mean(residuals_test):.6f}")
print(f"  Std: {np.std(residuals_test):.6f}")
print(f"  Skew: {pd.Series(residuals_test).skew():.6f}")
print(f"  Kurtosis: {pd.Series(residuals_test).kurtosis():.6f}")

# Identify worst predictions
test_results_df = test_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
test_results_df['predicted'] = y_pred_test
test_results_df['residual'] = residuals_test
test_results_df['abs_error'] = np.abs(residuals_test)
test_results_df['pct_error'] = np.abs(residuals_test / y_test) * 100

worst_predictions = test_results_df.nlargest(10, 'abs_error')
print("\nTop 10 Worst Predictions (by absolute error):")
print(worst_predictions[['Country', 'industry_code', 'carbon_intensity', 'predicted', 'abs_error']].to_string())

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING PREDICTIONS")
print("-" * 40)

# Save test predictions
test_results_df.to_csv('../results/svr_test_predictions.csv', index=False)
print("Test predictions saved to: ../results/svr_test_predictions.csv")

# Save train predictions
train_results_df = train_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
train_results_df['predicted'] = y_pred_train
train_results_df['residual'] = y_train - y_pred_train
train_results_df['abs_error'] = np.abs(train_results_df['residual'])
train_results_df.to_csv('../results/svr_train_predictions.csv', index=False)
print("Train predictions saved to: ../results/svr_train_predictions.csv")

# Save validation predictions
val_results_df = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
val_results_df['predicted'] = y_pred_val
val_results_df['residual'] = y_val - y_pred_val
val_results_df['abs_error'] = np.abs(val_results_df['residual'])
val_results_df.to_csv('../results/svr_val_predictions.csv', index=False)
print("Validation predictions saved to: ../results/svr_val_predictions.csv")

# Summary statistics
summary_stats = pd.DataFrame({
    'dataset': ['Train', 'Validation', 'Test'],
    'count': [len(y_train), len(y_val), len(y_test)],
    'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
    'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
    'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
    'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']]
})
summary_stats.to_csv('../results/svr_summary_stats.csv', index=False)
print("Summary statistics saved to: ../results/svr_summary_stats.csv")

# ============================================================================
# INDUSTRY-SPECIFIC ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("INDUSTRY-SPECIFIC ANALYSIS")
print("-" * 40)

industry_metrics = []
for industry in test_results_df['industry_code'].unique():
    mask = test_results_df['industry_code'] == industry
    if mask.sum() > 5:
        industry_results = evaluate_predictions(
            test_results_df.loc[mask, 'carbon_intensity'].values,
            test_results_df.loc[mask, 'predicted'].values,
            f"Industry {industry}"
        )
        industry_results['industry_code'] = industry
        industry_results['industry_name'] = test_results_df.loc[mask, 'industry_name'].iloc[0]
        industry_results['sample_size'] = mask.sum()
        industry_metrics.append(industry_results)

if industry_metrics:
    industry_df = pd.DataFrame(industry_metrics)
    industry_df = industry_df.sort_values('RMSE', ascending=False)
    industry_df.to_csv('../results/svr_industry_metrics.csv', index=False)
    print(f"Industry metrics saved to: ../results/svr_industry_metrics.csv")
    
    # Plot top/bottom industries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top_worst = industry_df.nlargest(10, 'RMSE')
    axes[0].barh(range(len(top_worst)), top_worst['RMSE'].values)
    axes[0].set_yticks(range(len(top_worst)))
    axes[0].set_yticklabels(top_worst['industry_name'].str[:30] + '...')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('SVR: Top 10 Industries with Highest RMSE')
    axes[0].invert_yaxis()
    
    top_best = industry_df.nsmallest(10, 'RMSE')
    axes[1].barh(range(len(top_best)), top_best['RMSE'].values)
    axes[1].set_yticks(range(len(top_best)))
    axes[1].set_yticklabels(top_best['industry_name'].str[:30] + '...')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('SVR: Top 10 Industries with Lowest RMSE')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../results/plots/svr_industry_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# COUNTRY-SPECIFIC ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("COUNTRY-SPECIFIC ANALYSIS")
print("-" * 40)

country_metrics = []
for country in test_results_df['Country Code'].unique():
    mask = test_results_df['Country Code'] == country
    if mask.sum() > 5:
        country_results = evaluate_predictions(
            test_results_df.loc[mask, 'carbon_intensity'].values,
            test_results_df.loc[mask, 'predicted'].values,
            f"Country {country}"
        )
        country_results['Country Code'] = country
        country_results['Country'] = test_results_df.loc[mask, 'Country'].iloc[0]
        country_results['sample_size'] = mask.sum()
        country_metrics.append(country_results)

if country_metrics:
    country_df = pd.DataFrame(country_metrics)
    country_df = country_df.sort_values('RMSE', ascending=False)
    country_df.to_csv('../results/svr_country_metrics.csv', index=False)
    print(f"Country metrics saved to: ../results/svr_country_metrics.csv")
    
    # Plot top/bottom countries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top_worst_countries = country_df.nlargest(10, 'RMSE')
    axes[0].barh(range(len(top_worst_countries)), top_worst_countries['RMSE'].values)
    axes[0].set_yticks(range(len(top_worst_countries)))
    axes[0].set_yticklabels(top_worst_countries['Country'].str[:30] + '...')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('SVR: Top 10 Countries with Highest RMSE')
    axes[0].invert_yaxis()
    
    top_best_countries = country_df.nsmallest(10, 'RMSE')
    axes[1].barh(range(len(top_best_countries)), top_best_countries['RMSE'].values)
    axes[1].set_yticks(range(len(top_best_countries)))
    axes[1].set_yticklabels(top_best_countries['Country'].str[:30] + '...')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('SVR: Top 10 Countries with Lowest RMSE')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../results/plots/svr_country_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predicted vs Actual - Test Set
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Carbon Intensity')
axes[0, 0].set_ylabel('Predicted Carbon Intensity')
axes[0, 0].set_title(f'SVR: Predicted vs Actual (R² = {test_metrics["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution
axes[0, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals_test), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(residuals_test):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('SVR: Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Learning Curve (Training progress - SVR doesn't have built-in learning curves)
axes[0, 2].text(0.5, 0.5, 'SVR does not provide learning curves\nlike tree-based models',
                ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
axes[0, 2].set_title('SVR Training Note')

# 4. Absolute Error Distribution
axes[1, 0].hist(np.abs(residuals_test), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=test_metrics['MAE'], color='r', linestyle='--', lw=2, 
                   label=f'MAE: {test_metrics["MAE"]:.4f}')
axes[1, 0].set_xlabel('Absolute Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('SVR: Absolute Error Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Metrics Comparison (Train vs Val vs Test)
metrics_compare = pd.DataFrame({
    'Train': [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['R2']],
    'Validation': [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['R2']],
    'Test': [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['R2']]
}, index=['MAE', 'RMSE', 'R²'])

metrics_compare.T.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('SVR: Metrics Across Datasets')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

# 6. Predicted vs Actual - Log Scale
axes[1, 2].scatter(y_test + 0.001, y_pred_test + 0.001, alpha=0.3, s=10)
axes[1, 2].set_xscale('log')
axes[1, 2].set_yscale('log')
axes[1, 2].plot([y_test.min()+0.001, y_test.max()+0.001], 
                [y_test.min()+0.001, y_test.max()+0.001], 'r--', lw=2)
axes[1, 2].set_xlabel('Actual (log scale)')
axes[1, 2].set_ylabel('Predicted (log scale)')
axes[1, 2].set_title('SVR: Log Scale Comparison')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/svr_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# ADDITIONAL SVR-SPECIFIC VISUALIZATION: Parameter Sensitivity
# ============================================================================

if len(results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # C parameter sensitivity
    c_sensitivity = pd.DataFrame(results).groupby('C')['val_rmse'].mean().reset_index()
    axes[0].plot(c_sensitivity['C'], c_sensitivity['val_rmse'], 'bo-')
    axes[0].set_xlabel('C Parameter (log scale)')
    axes[0].set_ylabel('Average Validation RMSE')
    axes[0].set_title('SVR: Sensitivity to C Parameter')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Epsilon sensitivity
    eps_sensitivity = pd.DataFrame(results).groupby('epsilon')['val_rmse'].mean().reset_index()
    axes[1].plot(eps_sensitivity['epsilon'], eps_sensitivity['val_rmse'], 'ro-')
    axes[1].set_xlabel('Epsilon Parameter')
    axes[1].set_ylabel('Average Validation RMSE')
    axes[1].set_title('SVR: Sensitivity to Epsilon Parameter')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/plots/svr_parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# SAVE ALL METRICS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING ALL METRICS")
print("-" * 40)

all_metrics = {
    'best_params': {k: str(v) if isinstance(v, (np.generic)) else v for k, v in best_params.items()},
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'tuning_time_seconds': tuning_time,
    'final_train_time_seconds': final_train_time,
    'total_combinations_tested': len(results),
    'tuning_strategy': 'sequential_search_on_subset',
    'tuning_subset_size': TUNE_SAMPLE_SIZE,
    'full_train_size': len(X_train_scaled)
}

with open('../results/svr_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("All metrics saved to: ../results/svr_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("SUPPORT VECTOR REGRESSION - FINAL SUMMARY")
print("=" * 60)
print(f"Best Parameters (found on {TUNE_SAMPLE_SIZE:,} sample):")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\nTotal combinations tested: {len(results)} (on subset)")
print(f"Tuning time: {tuning_time:.2f} seconds ({tuning_time/60:.2f} minutes)")
print(f"Final model training time (full {len(X_train_scaled):,} data): {final_train_time:.2f} seconds ({final_train_time/60:.2f} minutes)")
print("\n" + "-" * 40)
print("TEST SET PERFORMANCE")
print("-" * 40)
print(f"MAE:   {test_metrics['MAE']:.6f}")
print(f"RMSE:  {test_metrics['RMSE']:.6f}")
print(f"R²:    {test_metrics['R2']:.6f}")
print(f"MAPE:  {test_metrics['MAPE']:.2f}%")
print(f"SMAPE: {test_metrics['SMAPE']:.2f}%")
print(f"MedAE: {test_metrics['MedAE']:.6f}")
print("=" * 60)
print(f"Results saved to: ../results/")
print(f"Plots saved to: ../results/plots/")
print("=" * 60)