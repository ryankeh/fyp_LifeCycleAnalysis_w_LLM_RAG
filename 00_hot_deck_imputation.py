"""
Script: Baseline Hot-Deck Imputation for Carbon Intensity Prediction
Uses: train.csv, val.csv, test.csv
Saves: hotdeck_results.json, hotdeck_predictions.csv, and visualization plots
Based on methodology from "A global Water Quality Index and hot-deck imputation of missing data"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/plots").mkdir(parents=True, exist_ok=True)
Path("../results/plots/hotdeck").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("HOT-DECK IMPUTATION - BASELINE BENCHMARK")
print("=" * 60)

# ============================================================================
# DEFINE METRICS FUNCTIONS (Matching Random Forest metrics)
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
    """Calculate all metrics matching random forest evaluation"""
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
    print(f"  MedAE: {metrics['MedAE']:.6f}")
    print(f"  Max_Error: {metrics['Max_Error']:.6f}")
    
    return metrics

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

# Load feature info (THIS IS KEY - use same as random forest)
with open('../data/feature_info.json', 'r') as f:
    feature_info = json.load(f)

feature_cols = feature_info['feature_names']  # ALL features (should be 20)
target_col = feature_info['target_name']

print(f"\nTotal features from feature_info.json: {len(feature_cols)}")
print(f"First 10 features:")
for i, col in enumerate(feature_cols[:10]):
    print(f"  {i+1}. {col}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "-" * 40)
print("DATA PREPARATION")
print("-" * 40)

# Combine train and val for donor pool
donor_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
print(f"Donor pool size: {len(donor_df):,} rows")

# Recipient pool is test set
recipient_df = test_df.copy()
print(f"Recipient pool size: {len(recipient_df):,} rows")

# Use ALL features from feature_info.json for matching
matching_features = feature_cols
print(f"Matching features: {len(matching_features)}")

# Verify all features exist
missing_features = [f for f in matching_features if f not in donor_df.columns]
if missing_features:
    print(f"⚠️ Warning: Missing features: {missing_features[:5]}...")
    matching_features = [f for f in matching_features if f in donor_df.columns]
    print(f"Using {len(matching_features)} available features for matching")
else:
    print(f"✓ All {len(matching_features)} features available for matching")

# ============================================================================
# CORRELATION ANALYSIS FOR FALLBACK PRIORITIZATION
# ============================================================================

print("\n" + "-" * 40)
print("CORRELATION ANALYSIS (Feature Importance)")
print("-" * 40)

# Calculate correlations with target for all features
correlations = []
for feature in matching_features:
    if feature in donor_df.columns and feature != target_col:
        corr = donor_df[feature].corr(donor_df[target_col])
        correlations.append({
            'feature': feature,
            'correlation': abs(corr),
            'raw_correlation': corr
        })

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
print(f"\nTop 10 features by correlation with {target_col}:")
print(corr_df.head(10).to_string(index=False))

# Save correlation results
corr_df.to_csv('../results/hotdeck_feature_correlations.csv', index=False)
print("\nFeature correlations saved to: ../results/hotdeck_feature_correlations.csv")

# Create correlation visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Absolute correlation bar chart
ax1 = axes[0]
top_features = corr_df.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['raw_correlation']]
bars = ax1.barh(range(len(top_features)), top_features['correlation'].values, color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
ax1.set_xlabel('Absolute Correlation with Carbon Intensity')
ax1.set_title('Feature Importance for Matching (Higher = More Important)')
ax1.invert_yaxis()
ax1.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Low importance threshold')
ax1.legend()

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax1.text(row['correlation'] + 0.01, i, f'{row["correlation"]:.3f}', va='center', fontsize=8)

# Plot 2: Correlation heatmap of top features
ax2 = axes[1]
top_corr_features = top_features.head(10)['feature'].tolist() + [target_col]
corr_matrix = donor_df[top_corr_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, ax=ax2, square=True, cbar_kws={'shrink': 0.8})
ax2.set_title('Correlation Matrix: Top Features vs Target')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../results/plots/hotdeck/01_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print("Correlation plot saved to: ../results/plots/hotdeck/01_feature_correlations.png")

# Create fallback order (most to least important)
feature_importance_order = corr_df['feature'].tolist()
print(f"\nFallback order created: will drop least important features first")

# ============================================================================
# DISCRETIZATION OF CONTINUOUS FEATURES
# ============================================================================

print("\n" + "-" * 40)
print("FEATURE DISCRETIZATION")
print("-" * 40)

def discretize_series(series, n_bins=4):
    """Discretize continuous series into quartile bins"""
    # Check if already categorical/discrete
    if series.nunique() <= 10:
        return series.astype(str)
    
    # For continuous features, use quantile-based discretization
    try:
        labels = [f'Q{i+1}' for i in range(n_bins)]
        return pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    except:
        # Fallback to equal-width bins if quantiles fail
        return pd.cut(series, bins=n_bins, labels=[f'B{i+1}' for i in range(n_bins)])

# Create discretization mapping for each feature
discretization_info = {}
for feature in matching_features:
    if feature in donor_df.columns:
        donor_values = donor_df[feature].dropna()
        if len(donor_values) > 0:
            if donor_values.nunique() <= 10:
                # Already categorical-like
                discretization_info[feature] = {'type': 'categorical', 'values': donor_values.unique()}
            else:
                # Create bins based on percentiles
                percentiles = np.percentile(donor_values, [25, 50, 75])
                bins = [-np.inf] + percentiles.tolist() + [np.inf]
                labels = ['Q1', 'Q2', 'Q3', 'Q4']
                discretization_info[feature] = {
                    'type': 'continuous',
                    'bins': bins,
                    'labels': labels
                }

print(f"Discretization prepared for {len(discretization_info)} features")

def get_discretized_value(value, feature):
    """Get discretized category for a value"""
    if feature not in discretization_info:
        return str(value)
    
    info = discretization_info[feature]
    if info['type'] == 'categorical':
        return str(value)
    else:
        # Continuous feature with bins
        for i, (bin_low, bin_high) in enumerate(zip(info['bins'][:-1], info['bins'][1:])):
            if bin_low <= value <= bin_high:
                return info['labels'][i]
        return info['labels'][-1]  # Fallback to last bin

# Create discretized profiles
print("\nCreating donor profiles...")
donor_profiles = []
for idx, row in donor_df.iterrows():
    profile = {}
    for feature in matching_features:
        if feature in row and pd.notna(row[feature]):
            profile[feature] = get_discretized_value(row[feature], feature)
        else:
            profile[feature] = 'missing'
    donor_profiles.append({
        'index': idx,
        'profile': profile,
        'carbon_intensity': row[target_col]
    })

print(f"Creating recipient profiles...")
recipient_profiles = []
for idx, row in recipient_df.iterrows():
    profile = {}
    for feature in matching_features:
        if feature in row and pd.notna(row[feature]):
            profile[feature] = get_discretized_value(row[feature], feature)
        else:
            profile[feature] = 'missing'
    recipient_profiles.append({
        'index': idx,
        'profile': profile,
        'true_intensity': row[target_col] if target_col in row else None
    })

print(f"Created profiles for {len(donor_profiles)} donors and {len(recipient_profiles)} recipients")

# ============================================================================
# HOT-DECK MATCHING WITH PROGRESSIVE FALLBACK
# ============================================================================

print("\n" + "-" * 40)
print("HOT-DECK MATCHING (Progressive Fallback)")
print("-" * 40)

def find_matches(recipient_profile, donor_profiles, feature_order):
    """
    Find donors matching recipient profile with progressive fallback.
    Returns matched donor intensity and number of features matched.
    """
    # Try matching with progressively fewer features
    for n_features in range(len(feature_order), 0, -1):
        features_to_match = feature_order[:n_features]
        
        matches = []
        for donor in donor_profiles:
            match = True
            for feature in features_to_match:
                donor_val = donor['profile'].get(feature, 'missing')
                recipient_val = recipient_profile.get(feature, 'missing')
                if donor_val != recipient_val:
                    match = False
                    break
            if match:
                matches.append(donor)
        
        if matches:
            # Randomly select one match
            selected = np.random.choice(len(matches))
            return matches[selected]['carbon_intensity'], n_features
    
    # Ultimate fallback: random donor from entire pool
    selected = np.random.choice(len(donor_profiles))
    return donor_profiles[selected]['carbon_intensity'], 0

print("Matching recipients to donors...")
print(f"Starting with {len(feature_importance_order)} features...")

imputed_values = []
matched_features_count = []

for i, recipient in enumerate(recipient_profiles):
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(recipient_profiles)} recipients...")
    
    imputed_value, n_matched = find_matches(
        recipient['profile'], 
        donor_profiles, 
        feature_importance_order
    )
    imputed_values.append(imputed_value)
    matched_features_count.append(n_matched)

# Store results
recipient_df['hotdeck_imputed'] = imputed_values
recipient_df['matched_features'] = matched_features_count

print(f"\n✓ Matching complete!")
print(f"  Average features matched: {np.mean(matched_features_count):.2f}")
print(f"  Median features matched: {np.median(matched_features_count):.0f}")
print(f"  Min features matched: {np.min(matched_features_count)}")
print(f"  Max features matched: {np.max(matched_features_count)}")

# Match quality statistics
exact_matches = np.sum(np.array(matched_features_count) == len(feature_importance_order))
global_fallbacks = np.sum(np.array(matched_features_count) == 0)
print(f"\nMatch Quality:")
print(f"  Exact matches (all {len(feature_importance_order)} features): {exact_matches} ({exact_matches/len(matched_features_count)*100:.1f}%)")
print(f"  Global fallbacks (0 features): {global_fallbacks} ({global_fallbacks/len(matched_features_count)*100:.1f}%)")

# ============================================================================
# EVALUATION (Using same metrics as Random Forest)
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATION")
print("-" * 40)

# Get true values
true_values = recipient_df[target_col].values
predicted_values = recipient_df['hotdeck_imputed'].values

# Remove any rows with missing true values
valid_mask = ~np.isnan(true_values)
if valid_mask.sum() < len(true_values):
    print(f"⚠️ Removing {len(true_values) - valid_mask.sum()} rows with missing true values")
    true_values = true_values[valid_mask]
    predicted_values = predicted_values[valid_mask]
    recipient_df_eval = recipient_df[valid_mask].copy()
else:
    recipient_df_eval = recipient_df.copy()

metrics = evaluate_predictions(true_values, predicted_values, "HOT-DECK IMPUTATION")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

# Visualization 1: Fallback level distribution
fig, ax = plt.subplots(figsize=(12, 6))
fallback_counts = pd.Series(matched_features_count).value_counts().sort_index()
bars = ax.bar(range(len(fallback_counts)), fallback_counts.values)
ax.set_xticks(range(len(fallback_counts)))
ax.set_xticklabels([f"{x} features" if x > 0 else "0 features\n(global fallback)" for x in fallback_counts.index], 
                   rotation=45, ha='right')
ax.set_xlabel('Number of Features Matched')
ax.set_ylabel('Number of Imputed Values')
ax.set_title(f'Hot-Deck Imputation: Match Quality Distribution\n'
             f'Average: {np.mean(matched_features_count):.1f} features, '
             f'Total Features: {len(feature_importance_order)}')
for bar, val in zip(bars, fallback_counts.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:,}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/plots/hotdeck/02_fallback_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Predicted vs Actual
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(true_values, predicted_values, alpha=0.4, s=20, edgecolors='none')
min_val = min(true_values.min(), predicted_values.min())
max_val = max(true_values.max(), predicted_values.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Carbon Intensity')
ax.set_ylabel('Predicted Carbon Intensity')
ax.set_title(f'Hot-Deck Imputation: Predicted vs Actual\n'
             f'R² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/plots/hotdeck/03_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 3: Residual distribution and Q-Q plot
residuals = true_values - predicted_values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
ax1.axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, 
            label=f'Mean: {np.mean(residuals):.4f}')
ax1.set_xlabel('Residual (Actual - Predicted)')
ax1.set_ylabel('Frequency')
ax1.set_title('Residual Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normality Check)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/hotdeck/04_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 4: Distribution preservation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
donor_intensity = donor_df[target_col].values
ax1.hist(donor_intensity, bins=50, alpha=0.5, label='Donor Pool (Train+Val)', density=True, edgecolor='black')
ax1.hist(predicted_values, bins=50, alpha=0.5, label='Imputed Values', density=True, edgecolor='black')
ax1.set_xlabel('Carbon Intensity')
ax1.set_ylabel('Density')
ax1.set_title('Distribution Preservation Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
data_to_plot = [donor_intensity, predicted_values]
bp = ax2.boxplot(data_to_plot, labels=['Donor Pool\n(Actual)', 'Imputed\nValues'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
ax2.set_ylabel('Carbon Intensity')
ax2.set_title('Distribution Statistics Comparison')

stats_text = f"Donor Mean: {np.mean(donor_intensity):.4f}\n"
stats_text += f"Imputed Mean: {np.mean(predicted_values):.4f}\n"
stats_text += f"Donor Std: {np.std(donor_intensity):.4f}\n"
stats_text += f"Imputed Std: {np.std(predicted_values):.4f}"
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
         verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../results/plots/hotdeck/05_distribution_preservation.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 5: Error by match quality
fig, ax = plt.subplots(figsize=(12, 6))

# Group by number of features matched
match_groups = recipient_df_eval.groupby('matched_features').agg({
    'hotdeck_imputed': 'count',
    target_col: ['mean', 'std']
}).reset_index()
match_groups.columns = ['matched_features', 'count', 'actual_mean', 'actual_std']

# Calculate errors per group
errors = []
for n_features in match_groups['matched_features']:
    mask = recipient_df_eval['matched_features'] == n_features
    y_true_group = recipient_df_eval.loc[mask, target_col].values
    y_pred_group = recipient_df_eval.loc[mask, 'hotdeck_imputed'].values
    errors.append({
        'matched_features': n_features,
        'count': mask.sum(),
        'mae': mean_absolute_error(y_true_group, y_pred_group),
        'rmse': rmse(y_true_group, y_pred_group)
    })

error_df = pd.DataFrame(errors).sort_values('matched_features')

# Plot
x = error_df['matched_features']
width = 0.35
ax.bar(x - width/2, error_df['mae'], width, label='MAE', alpha=0.7)
ax.bar(x + width/2, error_df['rmse'], width, label='RMSE', alpha=0.7)
ax.set_xlabel('Number of Features Matched')
ax.set_ylabel('Error')
ax.set_title('Imputation Error by Match Quality')
ax.legend()
ax.grid(True, alpha=0.3)

# Add sample size annotations
for i, row in error_df.iterrows():
    ax.annotate(f'n={row["count"]}', 
                (row['matched_features'], max(row['mae'], row['rmse']) + 0.05),
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('../results/plots/hotdeck/06_error_by_match_quality.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAll visualizations saved to: ../results/plots/hotdeck/")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING RESULTS")
print("-" * 40)

# Save predictions
output_df = recipient_df[['Country', 'Country Code', 'industry_code', 'industry_name', target_col, 'hotdeck_imputed', 'matched_features']].copy()
output_df.rename(columns={target_col: 'actual_intensity'}, inplace=True)
output_df['residual'] = output_df['actual_intensity'] - output_df['hotdeck_imputed']
output_df['abs_error'] = np.abs(output_df['residual'])
output_df['pct_error'] = np.abs(output_df['residual'] / output_df['actual_intensity']) * 100
output_df.to_csv('../results/hotdeck_predictions.csv', index=False)
print("Predictions saved to: ../results/hotdeck_predictions.csv")

# Save metrics (matching random forest format)
all_metrics = {
    'method': 'Hot-Deck Imputation',
    'donor_pool_size': len(donor_df),
    'recipient_pool_size': len(recipient_df),
    'total_features': len(feature_cols),
    'features_used_for_matching': len(matching_features),
    'feature_correlations': corr_df.to_dict('records'),
    'match_statistics': {
        'avg_features_matched': float(np.mean(matched_features_count)),
        'median_features_matched': float(np.median(matched_features_count)),
        'std_features_matched': float(np.std(matched_features_count)),
        'min_features_matched': int(np.min(matched_features_count)),
        'max_features_matched': int(np.max(matched_features_count)),
        'exact_matches_pct': float(exact_matches / len(matched_features_count) * 100),
        'global_fallback_pct': float(global_fallbacks / len(matched_features_count) * 100)
    },
    'metrics': {
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
        'MAPE': metrics['MAPE'],
        'SMAPE': metrics['SMAPE'],
        'MedAE': metrics['MedAE'],
        'Max_Error': metrics['Max_Error']
    }
}

with open('../results/hotdeck_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("Metrics saved to: ../results/hotdeck_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("HOT-DECK IMPUTATION - SUMMARY")
print("=" * 60)
print(f"Donor pool: {len(donor_df):,} rows")
print(f"Recipients: {len(recipient_df):,} rows")
print(f"Total features in dataset: {len(feature_cols)}")
print(f"Features used for matching: {len(matching_features)}")
print("\nMatch Quality:")
print(f"  Average features matched: {np.mean(matched_features_count):.2f}")
print(f"  Median features matched: {np.median(matched_features_count):.0f}")
print(f"  Min features matched: {np.min(matched_features_count)}")
print(f"  Max features matched: {np.max(matched_features_count)}")
print(f"  Exact matches (all features): {exact_matches:,} ({exact_matches/len(matched_features_count)*100:.1f}%)")
print(f"  Global fallbacks: {global_fallbacks:,} ({global_fallbacks/len(matched_features_count)*100:.1f}%)")
print("\nTest Set Performance (Same metrics as Random Forest):")
print(f"  MAE:  {metrics['MAE']:.6f}")
print(f"  RMSE: {metrics['RMSE']:.6f}")
print(f"  R²:   {metrics['R2']:.6f}")
print(f"  MAPE: {metrics['MAPE']:.2f}%")
print(f"  SMAPE: {metrics['SMAPE']:.2f}%")
print(f"  MedAE: {metrics['MedAE']:.6f}")
print(f"  Max_Error: {metrics['Max_Error']:.6f}")
print("=" * 60)
print(f"\nResults saved to: ../results/")
print(f"Plots saved to: ../results/plots/hotdeck/")
print("=" * 60)