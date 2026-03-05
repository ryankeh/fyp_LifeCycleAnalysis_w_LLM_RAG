"""
Script 3: Evaluate model on validation set
Uses: val.csv, best_model.pt, scaler.pkl
Saves: evaluation_results.json, evaluation_plots.png
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/plots").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("EVALUATION SCRIPT")
print("=" * 60)
print(f"Using device: {device}")

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

def evaluate_predictions(y_true, y_pred):
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
    return metrics

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA AND MODEL")
print("-" * 40)

# Load validation data
val_df = pd.read_csv("../data/val.csv")
print(f"Validation set: {len(val_df):,} rows")

# Load feature info
with open('../data/feature_info.json', 'r') as f:
    feature_info = json.load(f)

feature_cols = feature_info['feature_names']
target_col = feature_info['target_name']

# Prepare data
X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df[target_col].values.astype(np.float32)

print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

# Load scaler
scaler = joblib.load('../models/scaler.pkl')
X_val_scaled = scaler.transform(X_val)

# Load target scaler
target_scaler = joblib.load('../models/target_scaler.pkl')

# Convert to tensors
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

# ============================================================================
# FIX: Load model with weights_only=False (safe since it's your own file)
# ============================================================================

print("\n" + "-" * 40)
print("LOADING MODEL")
print("-" * 40)

# Define the model architecture (must match training)
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Load checkpoint with weights_only=False (safe since it's your own file)
checkpoint = torch.load(
    '../models/best_model.pt', 
    map_location=device,
    weights_only=False  # Add this parameter to fix the error
)

model = SimpleNN(input_dim=len(feature_cols)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from epoch {checkpoint.get('epoch', 0) + 1}")
print(f"Validation loss at save: {checkpoint.get('val_loss', 'N/A')}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n" + "-" * 40)
print("GENERATING PREDICTIONS")
print("-" * 40)

with torch.no_grad():
    y_pred_scaled = model(X_val_tensor).cpu().numpy().flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

print(f"Predictions shape: {y_pred.shape}")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n" + "-" * 40)
print("CALCULATING METRICS")
print("-" * 40)

metrics = evaluate_predictions(y_val, y_pred)

print(f"\n{'='*50}")
print("EVALUATION RESULTS")
print('='*50)
print(f"MAE:  {metrics['MAE']:.6f}")
print(f"RMSE: {metrics['RMSE']:.6f}")
print(f"R²:   {metrics['R2']:.6f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"SMAPE: {metrics['SMAPE']:.2f}%")
print(f"MedAE: {metrics['MedAE']:.6f}")
print(f"Max Error: {metrics['Max_Error']:.6f}")
print('='*50)

# Calculate additional statistics
residuals = y_val - y_pred
print(f"\nResidual Statistics:")
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Std: {np.std(residuals):.6f}")
print(f"  Skew: {pd.Series(residuals).skew():.6f}")
print(f"  Kurtosis: {pd.Series(residuals).kurtosis():.6f}")

# ============================================================================
# SAVE METRICS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING RESULTS")
print("-" * 40)

# Save metrics to JSON
metrics['sample_size'] = len(y_val)
metrics['timestamp'] = pd.Timestamp.now().isoformat()
metrics['feature_count'] = len(feature_cols)

with open('../results/evaluation_results.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

print("Metrics saved to: ../results/evaluation_results.json")

# Save predictions for later analysis
results_df = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
results_df['predicted_intensity'] = y_pred
results_df['residual'] = residuals
results_df['absolute_error'] = np.abs(residuals)
results_df['percentage_error'] = np.abs(residuals / y_val) * 100
results_df.to_csv('../results/predictions.csv', index=False)
print("Predictions saved to: ../results/predictions.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_val, y_pred, alpha=0.3, s=10)
min_val = min(y_val.min(), y_pred.min())
max_val = max(y_val.max(), y_pred.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Carbon Intensity')
axes[0, 0].set_ylabel('Predicted Carbon Intensity')
axes[0, 0].set_title(f'Predicted vs Actual (R² = {metrics["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, label=f'Mean: {np.mean(residuals):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot (Normality Check)')
axes[0, 2].grid(True, alpha=0.3)

# 4. Residuals vs Predicted
axes[1, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Carbon Intensity')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Residuals vs Predicted')
axes[1, 0].grid(True, alpha=0.3)

# 5. Absolute Error Distribution
axes[1, 1].hist(np.abs(residuals), bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=metrics['MAE'], color='r', linestyle='--', lw=2, label=f'MAE: {metrics["MAE"]:.4f}')
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Absolute Error Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Metrics Bar Chart
metrics_to_plot = ['MAE', 'RMSE', 'R2', 'MAPE/100']
values_to_plot = [metrics['MAE'], metrics['RMSE'], metrics['R2'], metrics['MAPE']/100]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = axes[1, 2].bar(metrics_to_plot, values_to_plot, color=colors)
axes[1, 2].set_ylabel('Value')
axes[1, 2].set_title('Key Metrics')
for bar, val in zip(bars, values_to_plot):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../results/plots/evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# INDUSTRY-SPECIFIC ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("INDUSTRY-SPECIFIC ANALYSIS")
print("-" * 40)

industry_metrics = []
for industry in results_df['industry_code'].unique():
    mask = results_df['industry_code'] == industry
    if mask.sum() > 10:
        industry_results = evaluate_predictions(
            results_df.loc[mask, 'carbon_intensity'].values,
            results_df.loc[mask, 'predicted_intensity'].values
        )
        industry_results['industry_code'] = industry
        industry_results['industry_name'] = results_df.loc[mask, 'industry_name'].iloc[0]
        industry_results['sample_size'] = mask.sum()
        industry_metrics.append(industry_results)

if industry_metrics:
    industry_df = pd.DataFrame(industry_metrics)
    industry_df = industry_df.sort_values('RMSE', ascending=False)
    industry_df.to_csv('../results/industry_metrics.csv', index=False)
    print(f"Industry metrics saved to: ../results/industry_metrics.csv")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE - SUMMARY")
print("=" * 60)
print(f"Validation samples: {len(y_val):,}")
print(f"MAE:  {metrics['MAE']:.6f}")
print(f"RMSE: {metrics['RMSE']:.6f}")
print(f"R²:   {metrics['R2']:.6f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print("=" * 60)
print(f"Results saved to: ../results/")
print(f"Plots saved to: ../results/plots/")
print("=" * 60)