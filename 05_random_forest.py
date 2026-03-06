"""
Script 5: Random Forest for Carbon Intensity Prediction
Uses: train.csv, val.csv, test.csv
Saves: random_forest_results.json, random_forest_predictions.csv, visualizations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
Path("../models").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RANDOM FOREST REGRESSION SCRIPT")
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
# HYPERPARAMETER TUNING WITH VALIDATION SET
# ============================================================================

print("\n" + "-" * 40)
print("HYPERPARAMETER TUNING")
print("-" * 40)

# Define hyperparameters to test
n_estimators_list = [50, 100, 200]
max_depth_list = [10, 20, 30, None]
min_samples_split_list = [2, 5, 10]

best_val_rmse = float('inf')
best_params = {}
best_model = None
results = []

print("Testing different hyperparameter combinations...")

for n_est in n_estimators_list:
    for max_d in max_depth_list:
        for min_split in min_samples_split_list:
            # Train model
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_split,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            rf.fit(X_train, y_train)
            
            # Predict on validation
            y_pred_val = rf.predict(X_val)
            val_rmse = rmse(y_val, y_pred_val)
            
            # Store results
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d if max_d is not None else 'None',
                'min_samples_split': min_split,
                'val_rmse': val_rmse
            })
            
            # Update best model
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = {
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'min_samples_split': min_split
                }
                best_model = rf
            
            print(f"  n_est={n_est:3d}, max_depth={str(max_d):4s}, min_split={min_split:2d} -> RMSE: {val_rmse:.6f}")

print(f"\nBest parameters: {best_params}")
print(f"Best validation RMSE: {best_val_rmse:.6f}")

# Save tuning results
tuning_df = pd.DataFrame(results)
tuning_df.to_csv('../results/rf_tuning_results.csv', index=False)
print("Tuning results saved to: ../results/rf_tuning_results.csv")

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING FINAL MODEL")
print("-" * 40)

# Train on full training data with best params
final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42,
    n_jobs=-1,
    verbose=1
)

final_model.fit(X_train, y_train)
print("Final model trained!")

# Save model
joblib.dump(final_model, '../models/random_forest_best_model.pkl')
print("Model saved to: ../models/random_forest_best_model.pkl")

# ============================================================================
# PREDICTIONS ON ALL SETS
# ============================================================================

print("\n" + "-" * 40)
print("GENERATING PREDICTIONS")
print("-" * 40)

y_pred_train = final_model.predict(X_train)
y_pred_val = final_model.predict(X_val)
y_pred_test = final_model.predict(X_test)

print("Predictions generated for all sets")

# ============================================================================
# EVALUATE ON ALL SETS
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATION RESULTS")
print("-" * 40)

train_metrics = evaluate_predictions(y_train, y_pred_train, "TRAIN SET")
val_metrics = evaluate_predictions(y_val, y_pred_val, "VALIDATION SET")
test_metrics = evaluate_predictions(y_test, y_pred_test, "TEST SET")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Get feature importances
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('../results/rf_feature_importance.csv', index=False)
print("\nFeature importance saved to: ../results/rf_feature_importance.csv")

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
test_results_df.to_csv('../results/rf_test_predictions.csv', index=False)
print("Test predictions saved to: ../results/rf_test_predictions.csv")

# Save train predictions separately
train_results_df = train_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
train_results_df['predicted'] = y_pred_train
train_results_df['residual'] = y_train - y_pred_train
train_results_df['abs_error'] = np.abs(train_results_df['residual'])
train_results_df.to_csv('../results/rf_train_predictions.csv', index=False)
print("Train predictions saved to: ../results/rf_train_predictions.csv")

# Save validation predictions separately
val_results_df = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
val_results_df['predicted'] = y_pred_val
val_results_df['residual'] = y_val - y_pred_val
val_results_df['abs_error'] = np.abs(val_results_df['residual'])
val_results_df.to_csv('../results/rf_val_predictions.csv', index=False)
print("Validation predictions saved to: ../results/rf_val_predictions.csv")

# Summary statistics
summary_stats = pd.DataFrame({
    'dataset': ['Train', 'Validation', 'Test'],
    'count': [len(y_train), len(y_val), len(y_test)],
    'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
    'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
    'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
    'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']]
})
summary_stats.to_csv('../results/rf_summary_stats.csv', index=False)
print("Summary statistics saved to: ../results/rf_summary_stats.csv")

# ============================================================================
# INDUSTRY-SPECIFIC ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("INDUSTRY-SPECIFIC ANALYSIS")
print("-" * 40)

industry_metrics = []
for industry in test_results_df['industry_code'].unique():
    mask = test_results_df['industry_code'] == industry
    if mask.sum() > 5:  # Only analyze industries with at least 5 samples
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
    industry_df.to_csv('../results/rf_industry_metrics.csv', index=False)
    print(f"Industry metrics saved to: ../results/rf_industry_metrics.csv")
    
    # Plot top/bottom industries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10 worst performing industries (highest RMSE)
    top_worst = industry_df.nlargest(10, 'RMSE')
    axes[0].barh(range(len(top_worst)), top_worst['RMSE'].values)
    axes[0].set_yticks(range(len(top_worst)))
    axes[0].set_yticklabels(top_worst['industry_name'].str[:30] + '...')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Top 10 Industries with Highest RMSE')
    axes[0].invert_yaxis()
    
    # Top 10 best performing industries (lowest RMSE)
    top_best = industry_df.nsmallest(10, 'RMSE')
    axes[1].barh(range(len(top_best)), top_best['RMSE'].values)
    axes[1].set_yticks(range(len(top_best)))
    axes[1].set_yticklabels(top_best['industry_name'].str[:30] + '...')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Top 10 Industries with Lowest RMSE')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../results/plots/rf_industry_performance.png', dpi=150, bbox_inches='tight')
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
    if mask.sum() > 5:  # Only analyze countries with at least 5 samples
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
    country_df.to_csv('../results/rf_country_metrics.csv', index=False)
    print(f"Country metrics saved to: ../results/rf_country_metrics.csv")
    
    # Plot top/bottom countries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10 worst performing countries (highest RMSE)
    top_worst_countries = country_df.nlargest(10, 'RMSE')
    axes[0].barh(range(len(top_worst_countries)), top_worst_countries['RMSE'].values)
    axes[0].set_yticks(range(len(top_worst_countries)))
    axes[0].set_yticklabels(top_worst_countries['Country'].str[:30] + '...')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Top 10 Countries with Highest RMSE')
    axes[0].invert_yaxis()
    
    # Top 10 best performing countries (lowest RMSE)
    top_best_countries = country_df.nsmallest(10, 'RMSE')
    axes[1].barh(range(len(top_best_countries)), top_best_countries['RMSE'].values)
    axes[1].set_yticks(range(len(top_best_countries)))
    axes[1].set_yticklabels(top_best_countries['Country'].str[:30] + '...')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Top 10 Countries with Lowest RMSE')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../results/plots/rf_country_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Distribution of country RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(country_df['RMSE'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=country_df['RMSE'].median(), color='r', linestyle='--', 
               label=f'Median: {country_df["RMSE"].median():.4f}')
    ax.axvline(x=country_df['RMSE'].mean(), color='g', linestyle='--', 
               label=f'Mean: {country_df["RMSE"].mean():.4f}')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Number of Countries')
    ax.set_title('Distribution of Country-Level RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/plots/rf_country_rmse_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Sample size vs RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(country_df['sample_size'], country_df['RMSE'], 
                        alpha=0.6, c=country_df['RMSE'], cmap='viridis', s=50)
    ax.set_xlabel('Number of Samples per Country')
    ax.set_ylabel('RMSE')
    ax.set_title('Country Performance by Sample Size')
    plt.colorbar(scatter, ax=ax, label='RMSE')
    ax.grid(True, alpha=0.3)
    
    # Add labels for extreme points
    for idx, row in country_df.nlargest(5, 'RMSE').iterrows():
        ax.annotate(row['Country'], (row['sample_size'], row['RMSE']), 
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('../results/plots/rf_country_samples_vs_rmse.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# 1. Predicted vs Actual - Test Set
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Carbon Intensity')
axes[0, 0].set_ylabel('Predicted Carbon Intensity')
axes[0, 0].set_title(f'Test Set: Predicted vs Actual (R² = {test_metrics["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution - Test Set
axes[0, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals_test), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(residuals_test):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Test Set: Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot
from scipy import stats
stats.probplot(residuals_test, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Test Set: Q-Q Plot')
axes[0, 2].grid(True, alpha=0.3)

# 4. Residuals vs Predicted
axes[1, 0].scatter(y_pred_test, residuals_test, alpha=0.3, s=10)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Carbon Intensity')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Test Set: Residuals vs Predicted')
axes[1, 0].grid(True, alpha=0.3)

# 5. Absolute Error Distribution
axes[1, 1].hist(np.abs(residuals_test), bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=test_metrics['MAE'], color='r', linestyle='--', lw=2, 
                   label=f'MAE: {test_metrics["MAE"]:.4f}')
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Test Set: Absolute Error Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Metrics Comparison (Train vs Val vs Test)
metrics_compare = pd.DataFrame({
    'Train': [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['R2']],
    'Validation': [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['R2']],
    'Test': [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['R2']]
}, index=['MAE', 'RMSE', 'R²'])

metrics_compare.T.plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Metrics Comparison Across Datasets')
axes[1, 2].set_ylabel('Value')
axes[1, 2].legend(loc='upper right')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45)

# 7. Feature Importance (Top 15)
top_features = feature_importance_df.head(15)
axes[2, 0].barh(range(len(top_features)), top_features['importance'].values)
axes[2, 0].set_yticks(range(len(top_features)))
axes[2, 0].set_yticklabels(top_features['feature'].values)
axes[2, 0].set_xlabel('Importance')
axes[2, 0].set_title('Top 15 Feature Importances')
axes[2, 0].invert_yaxis()

# 8. Learning Curves (OOB Error vs Trees)
if hasattr(final_model, 'oob_score_'):
    oob_scores = []
    trees_range = range(10, best_params['n_estimators'] + 1, 10)
    for i in trees_range:
        rf_temp = RandomForestRegressor(
            n_estimators=i,
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        rf_temp.fit(X_train, y_train)
        oob_scores.append(rf_temp.oob_score_)
    
    axes[2, 1].plot(trees_range, oob_scores, 'b-')
    axes[2, 1].set_xlabel('Number of Trees')
    axes[2, 1].set_ylabel('OOB Score (R²)')
    axes[2, 1].set_title('Out-of-Bag Learning Curve')
    axes[2, 1].grid(True, alpha=0.3)
else:
    axes[2, 1].text(0.5, 0.5, 'OOB Score not available\n(need oob_score=True during training)', 
                    ha='center', va='center', transform=axes[2, 1].transAxes)

# 9. Predicted vs Actual - Log Scale (for small values)
axes[2, 2].scatter(y_test + 0.001, y_pred_test + 0.001, alpha=0.3, s=10)
axes[2, 2].set_xscale('log')
axes[2, 2].set_yscale('log')
axes[2, 2].plot([y_test.min()+0.001, y_test.max()+0.001], 
                [y_test.min()+0.001, y_test.max()+0.001], 'r--', lw=2)
axes[2, 2].set_xlabel('Actual (log scale)')
axes[2, 2].set_ylabel('Predicted (log scale)')
axes[2, 2].set_title('Test Set: Predicted vs Actual (Log Scale)')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/random_forest_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE ALL METRICS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING ALL METRICS")
print("-" * 40)

all_metrics = {
    'best_params': best_params,
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'feature_importance': feature_importance_df.to_dict('records')
}

with open('../results/random_forest_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("All metrics saved to: ../results/random_forest_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("RANDOM FOREST - FINAL SUMMARY")
print("=" * 60)
print(f"Best Parameters:")
print(f"  n_estimators: {best_params['n_estimators']}")
print(f"  max_depth: {best_params['max_depth']}")
print(f"  min_samples_split: {best_params['min_samples_split']}")
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