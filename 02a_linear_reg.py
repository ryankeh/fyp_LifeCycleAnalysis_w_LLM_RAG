"""
Script 4: Linear Regression for Carbon Intensity Prediction
Uses: train.csv, val.csv, test.csv
Saves: linear_regression_results.json, linear_regression_predictions.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/plots").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LINEAR REGRESSION SCRIPT")
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
# NORMALIZE FEATURES
# ============================================================================

print("\n" + "-" * 40)
print("NORMALIZING FEATURES")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, '../models/linear_scaler.pkl')
print("Scaler saved to: ../models/linear_scaler.pkl")

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
# TRAIN LINEAR REGRESSION MODELS
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING LINEAR REGRESSION MODELS")
print("-" * 40)

# 1. Ordinary Least Squares
print("\n1. Ordinary Least Squares (OLS)")
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_ols_train = ols_model.predict(X_train_scaled)
y_pred_ols_val = ols_model.predict(X_val_scaled)
y_pred_ols_test = ols_model.predict(X_test_scaled)

# Evaluate
ols_train_metrics = evaluate_predictions(y_train, y_pred_ols_train, "OLS Train")
ols_val_metrics = evaluate_predictions(y_val, y_pred_ols_val, "OLS Validation")
ols_test_metrics = evaluate_predictions(y_test, y_pred_ols_test, "OLS Test")

# 2. Ridge Regression (L2 regularization)
print("\n2. Ridge Regression (with cross-validation for alpha)")
from sklearn.linear_model import RidgeCV

# Try different alpha values
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {ridge_cv.alpha_:.4f}")

# Predictions
y_pred_ridge_train = ridge_cv.predict(X_train_scaled)
y_pred_ridge_val = ridge_cv.predict(X_val_scaled)
y_pred_ridge_test = ridge_cv.predict(X_test_scaled)

# Evaluate
ridge_train_metrics = evaluate_predictions(y_train, y_pred_ridge_train, "Ridge Train")
ridge_val_metrics = evaluate_predictions(y_val, y_pred_ridge_val, "Ridge Validation")
ridge_test_metrics = evaluate_predictions(y_test, y_pred_ridge_test, "Ridge Test")

# 3. Lasso Regression (L1 regularization)
print("\n3. Lasso Regression (with cross-validation for alpha)")
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {lasso_cv.alpha_:.4f}")
print(f"Number of features used: {np.sum(lasso_cv.coef_ != 0)}/{len(feature_cols)}")

# Predictions
y_pred_lasso_train = lasso_cv.predict(X_train_scaled)
y_pred_lasso_val = lasso_cv.predict(X_val_scaled)
y_pred_lasso_test = lasso_cv.predict(X_test_scaled)

# Evaluate
lasso_train_metrics = evaluate_predictions(y_train, y_pred_lasso_train, "Lasso Train")
lasso_val_metrics = evaluate_predictions(y_val, y_pred_lasso_val, "Lasso Validation")
lasso_test_metrics = evaluate_predictions(y_test, y_pred_lasso_test, "Lasso Test")

# ============================================================================
# COMPARE MODELS
# ============================================================================

print("\n" + "-" * 40)
print("MODEL COMPARISON ON VALIDATION SET")
print("-" * 40)

comparison_df = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso'],
    'MAE': [ols_val_metrics['MAE'], ridge_val_metrics['MAE'], lasso_val_metrics['MAE']],
    'RMSE': [ols_val_metrics['RMSE'], ridge_val_metrics['RMSE'], lasso_val_metrics['RMSE']],
    'R2': [ols_val_metrics['R2'], ridge_val_metrics['R2'], lasso_val_metrics['R2']],
    'MAPE': [ols_val_metrics['MAPE'], ridge_val_metrics['MAPE'], lasso_val_metrics['MAPE']]
})

print("\nComparison Table:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_idx = comparison_df['RMSE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\nBest model based on RMSE: {best_model_name}")

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n" + "-" * 40)
print("SAVING BEST MODEL")
print("-" * 40)

if best_model_name == 'OLS':
    best_model = ols_model
    best_pred_val = y_pred_ols_val
    best_pred_test = y_pred_ols_test
    best_metrics_val = ols_val_metrics
    best_metrics_test = ols_test_metrics
elif best_model_name == 'Ridge':
    best_model = ridge_cv
    best_pred_val = y_pred_ridge_val
    best_pred_test = y_pred_ridge_test
    best_metrics_val = ridge_val_metrics
    best_metrics_test = ridge_test_metrics
else:
    best_model = lasso_cv
    best_pred_val = y_pred_lasso_val
    best_pred_test = y_pred_lasso_test
    best_metrics_val = lasso_val_metrics
    best_metrics_test = lasso_test_metrics

# Save model
joblib.dump(best_model, f'../models/linear_best_model_{best_model_name.lower()}.pkl')
print(f"Best model saved to: ../models/linear_best_model_{best_model_name.lower()}.pkl")

# Save feature coefficients
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': best_model.coef_
})
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)
coef_df.to_csv('../results/linear_feature_coefficients.csv', index=False)
print("Feature coefficients saved to: ../results/linear_feature_coefficients.csv")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING PREDICTIONS")
print("-" * 40)

# Validation set predictions
val_results = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
val_results['predicted_ols'] = y_pred_ols_val
val_results['predicted_ridge'] = y_pred_ridge_val
val_results['predicted_lasso'] = y_pred_lasso_val
val_results['best_model'] = best_model_name
val_results['best_prediction'] = best_pred_val
val_results['residual'] = val_results['carbon_intensity'] - val_results['best_prediction']
val_results['abs_error'] = np.abs(val_results['residual'])
val_results.to_csv('../results/linear_predictions.csv', index=False)
print("Predictions saved to: ../results/linear_predictions.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Best Model: Predicted vs Actual
axes[0, 0].scatter(y_val, best_pred_val, alpha=0.3, s=10)
min_val = min(y_val.min(), best_pred_val.min())
max_val = max(y_val.max(), best_pred_val.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Carbon Intensity')
axes[0, 0].set_ylabel('Predicted Carbon Intensity')
axes[0, 0].set_title(f'{best_model_name} - Predicted vs Actual (R² = {best_metrics_val["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution
residuals = y_val - best_pred_val
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(residuals):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'{best_model_name} - Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Model Comparison - RMSE
models = ['OLS', 'Ridge', 'Lasso']
rmse_values = [ols_val_metrics['RMSE'], ridge_val_metrics['RMSE'], lasso_val_metrics['RMSE']]
bars = axes[0, 2].bar(models, rmse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 2].set_ylabel('RMSE')
axes[0, 2].set_title('Model Comparison - RMSE')
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')

# 4. Feature Importance (Top 10)
top_features = coef_df.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['coefficient'].values)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'].values)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title(f'Top 10 Feature Coefficients ({best_model_name})')
axes[1, 0].axvline(x=0, color='black', linestyle='-', lw=0.5)

# 5. Predicted vs Actual - Log Scale (to see small values better)
axes[1, 1].scatter(y_val + 0.001, best_pred_val + 0.001, alpha=0.3, s=10)  # Add small constant to avoid log(0)
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].plot([y_val.min()+0.001, y_val.max()+0.001], 
                [y_val.min()+0.001, y_val.max()+0.001], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual (log scale)')
axes[1, 1].set_ylabel('Predicted (log scale)')
axes[1, 1].set_title('Predicted vs Actual (Log Scale)')
axes[1, 1].grid(True, alpha=0.3)

# 6. Absolute Error Distribution
axes[1, 2].hist(np.abs(residuals), bins=50, edgecolor='black', alpha=0.7)
axes[1, 2].axvline(x=best_metrics_val['MAE'], color='r', linestyle='--', 
                   lw=2, label=f'MAE: {best_metrics_val["MAE"]:.4f}')
axes[1, 2].set_xlabel('Absolute Error')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title(f'{best_model_name} - Absolute Error Distribution')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE ALL METRICS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING METRICS")
print("-" * 40)

all_metrics = {
    'OLS': {
        'train': ols_train_metrics,
        'validation': ols_val_metrics,
        'test': ols_test_metrics
    },
    'Ridge': {
        'train': ridge_train_metrics,
        'validation': ridge_val_metrics,
        'test': ridge_test_metrics
    },
    'Lasso': {
        'train': lasso_train_metrics,
        'validation': lasso_val_metrics,
        'test': lasso_test_metrics
    },
    'best_model': best_model_name,
    'best_model_validation': best_metrics_val,
    'best_model_test': best_metrics_test
}

with open('../results/linear_regression_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("All metrics saved to: ../results/linear_regression_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION - SUMMARY")
print("=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Validation R²: {best_metrics_val['R2']:.4f}")
print(f"Validation MAE: {best_metrics_val['MAE']:.4f}")
print(f"Validation RMSE: {best_metrics_val['RMSE']:.4f}")
print(f"Validation MAPE: {best_metrics_val['MAPE']:.2f}%")
print("\nTest Set Performance:")
print(f"  R²: {best_metrics_test['R2']:.4f}")
print(f"  MAE: {best_metrics_test['MAE']:.4f}")
print(f"  RMSE: {best_metrics_test['RMSE']:.4f}")
print(f"  MAPE: {best_metrics_test['MAPE']:.2f}%")
print("=" * 60)