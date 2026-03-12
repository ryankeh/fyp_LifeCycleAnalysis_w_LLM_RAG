"""
Script 11b: Industry FFNN Only (No Country Variables/GCN) for Carbon Intensity Prediction
Ablation study: Removes GCN country encoder, uses only industry variables via FFNN
Uses: train.csv, val.csv, test.csv
Saves: industry_only_results.json, predictions, visualizations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("INDUSTRY FFNN ONLY (NO COUNTRY VARIABLES/GCN) - ABLATION STUDY")
print("=" * 60)
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# ABLATION 2: Industry FFNN Only Model (No GCN/Country Features)
# ============================================================================

class IndustryFFNNAblationPredictor(nn.Module):
    """
    Ablation model: Only industry variables via FFNN, no GCN/country features.
    Uses only industry embeddings to predict carbon intensity.
    """
    def __init__(self, num_industries, industry_variables,
                 industry_embedding_dim=32, final_hidden_dim=64, dropout=0.2):
        super(IndustryFFNNAblationPredictor, self).__init__()
        
        # Store industry variables as a buffer
        self.register_buffer('industry_vars', 
                           torch.FloatTensor(industry_variables))  # [num_industries × 8]
        
        # Learnable projection: 8 dimensions → 32 dimensions (same as original)
        self.projection = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, industry_embedding_dim)
        )
        
        # Optional: Small learnable residual (same as original)
        self.residual = nn.Embedding(num_industries, industry_embedding_dim)
        nn.init.zeros_(self.residual.weight)
        
        # Final prediction network (takes only industry embeddings)
        # To match original total parameters, we make final network wider
        # Original: country(32) + industry(32) = 64 input
        # Now: only industry(32) input
        self.predictor = nn.Sequential(
            nn.Linear(industry_embedding_dim, final_hidden_dim * 2),  # Wider to compensate
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim * 2, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, 1)
        )
        
    def get_industry_embedding(self, industry_idx):
        """
        Args:
            industry_idx: [batch_size] indices of industries
        Returns:
            industry_embeddings: [batch_size × 32]
        """
        # Get the 8 variables for these industries
        vars_batch = self.industry_vars[industry_idx]  # [batch × 8]
        
        # Project 8 → 32 using learned transformation
        projected = self.projection(vars_batch)  # [batch × 32]
        
        # Add small learnable residual (starts at 0, can adjust)
        residual = self.residual(industry_idx)  # [batch × 32]
        
        # Combine: expert knowledge + data-driven fine-tuning
        industry_embeddings = projected + 0.1 * residual  # Residual scaled down
        
        return industry_embeddings
        
    def forward(self, country_idx, industry_idx):
        """
        Args:
            country_idx: ignored (kept for API compatibility)
            industry_idx: [batch_size] indices of industries
        """
        # Get industry embeddings from your 8 variables
        industry_emb = self.get_industry_embedding(industry_idx)  # [batch × 32]
        
        # Predict directly from industry embeddings
        output = self.predictor(industry_emb)
        
        return output.squeeze()

# ============================================================================
# METRICS FUNCTIONS
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

# Industry variables only (your 8 scores)
industry_variables_cols = [col for col in feature_cols if col in
                          ['process_emission_intensity_score', 'material_processing_depth_score',
                           'thermal_process_intensity_score', 'electrification_feasibility_score',
                           'continuous_operations_intensity_score', 'material_throughput_scale_score',
                           'chemical_intensity_score', 'capital_vs_labor_intensity_score']]

print(f"\nIndustry variables ({len(industry_variables_cols)}):")
for col in industry_variables_cols:
    print(f"  {col}")

# Get unique countries and industries
countries = sorted(train_df['Country Code'].unique())
industries = sorted(train_df['industry_code'].unique())

# Create mappings
country_to_idx = {country: i for i, country in enumerate(countries)}
industry_to_idx = {industry: i for i, industry in enumerate(industries)}

num_countries = len(countries)
num_industries = len(industries)

print(f"\nNumber of countries: {num_countries}")
print(f"Number of industries: {num_industries}")

# ============================================================================
# PREPARE INDUSTRY VARIABLES (your 8 scores)
# ============================================================================

print("\n" + "-" * 40)
print("PREPARING INDUSTRY VARIABLES")
print("-" * 40)

# Create matrix of industry variables [num_industries × 8]
industry_vars_list = []
for industry in industries:
    # Take first occurrence (industry variables are constant across countries)
    industry_data = train_df[train_df['industry_code'] == industry][industry_variables_cols].iloc[0].values
    industry_vars_list.append(industry_data)

industry_variables = np.array(industry_vars_list, dtype=np.float32)

# Scale industry variables
industry_scaler = StandardScaler()
industry_variables_scaled = industry_scaler.fit_transform(industry_variables)

print(f"Industry variables shape: {industry_variables_scaled.shape}")
print(f"Industry variables stats:")
print(f"  Mean: {industry_variables_scaled.mean(axis=0).round(3)}")
print(f"  Std:  {industry_variables_scaled.std(axis=0).round(3)}")

# Save scaler for later use
joblib.dump(industry_scaler, '../models/industry_only_scaler.pkl')
print("Industry scaler saved to: ../models/industry_only_scaler.pkl")

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

print("\n" + "-" * 40)
print("PREPARING TRAINING DATA")
print("-" * 40)

# Map country codes and industry codes to indices
train_country_idx = torch.LongTensor([country_to_idx[c] for c in train_df['Country Code']])
train_industry_idx = torch.LongTensor([industry_to_idx[i] for i in train_df['industry_code']])
train_target = torch.FloatTensor(train_df[target_col].values)

val_country_idx = torch.LongTensor([country_to_idx[c] for c in val_df['Country Code']])
val_industry_idx = torch.LongTensor([industry_to_idx[i] for i in val_df['industry_code']])
val_target = torch.FloatTensor(val_df[target_col].values)

test_country_idx = torch.LongTensor([country_to_idx[c] for c in test_df['Country Code']])
test_industry_idx = torch.LongTensor([industry_to_idx[i] for i in test_df['industry_code']])
test_target = torch.FloatTensor(test_df[target_col].values)

print(f"Train samples: {len(train_target)}")
print(f"Validation samples: {len(val_target)}")
print(f"Test samples: {len(test_target)}")

# Create data loaders (country_idx is kept but ignored by model)
batch_size = 256

train_dataset = TensorDataset(train_country_idx, train_industry_idx, train_target)
val_dataset = TensorDataset(val_country_idx, val_industry_idx, val_target)
test_dataset = TensorDataset(test_country_idx, test_industry_idx, test_target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ============================================================================
# INITIALIZE MODEL (Industry FFNN Only)
# ============================================================================

print("\n" + "-" * 40)
print("INITIALIZING INDUSTRY-ONLY MODEL")
print("-" * 40)

model = IndustryFFNNAblationPredictor(
    num_industries=num_industries,
    industry_variables=industry_variables_scaled,
    industry_embedding_dim=32,
    final_hidden_dim=64,
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING SETUP")
print("-" * 40)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Early stopping
best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_epoch = 0

# Training history
history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}

num_epochs = 100
print(f"Max epochs: {num_epochs}")
print(f"Early stopping patience: {patience}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING LOOP")
print("-" * 40)

start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    
    for batch_country, batch_industry, batch_target in train_loader:
        batch_industry = batch_industry.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_country, batch_industry)  # Country idx ignored
        loss = criterion(predictions, batch_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    avg_train_loss = np.mean(train_losses)
    
    # Validation phase
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch_country, batch_industry, batch_target in val_loader:
            batch_industry = batch_industry.to(device)
            batch_target = batch_target.to(device)
            
            predictions = model(batch_country, batch_industry)
            loss = criterion(predictions, batch_target)
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Update scheduler
    scheduler.step(avg_val_loss)
    
    # Save history
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['lr'].append(current_lr)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.6f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_epoch = epoch + 1
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'industry_variables_cols': industry_variables_cols,
            'industries': industries,
            'industry_to_idx': industry_to_idx
        }, '../models/industry_only_best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n" + "-" * 40)
print("GENERATING PREDICTIONS")
print("-" * 40)

model.eval()
predictions = {'train': [], 'val': [], 'test': []}
targets = {'train': [], 'val': [], 'test': []}

with torch.no_grad():
    # Train predictions
    for batch_country, batch_industry, batch_target in train_loader:
        batch_industry = batch_industry.to(device)
        pred = model(batch_country, batch_industry)
        predictions['train'].extend(pred.cpu().numpy())
        targets['train'].extend(batch_target.numpy())
    
    # Validation predictions
    for batch_country, batch_industry, batch_target in val_loader:
        batch_industry = batch_industry.to(device)
        pred = model(batch_country, batch_industry)
        predictions['val'].extend(pred.cpu().numpy())
        targets['val'].extend(batch_target.numpy())
    
    # Test predictions
    for batch_country, batch_industry, batch_target in test_loader:
        batch_industry = batch_industry.to(device)
        pred = model(batch_country, batch_industry)
        predictions['test'].extend(pred.cpu().numpy())
        targets['test'].extend(batch_target.numpy())

y_pred_train = np.array(predictions['train'])
y_pred_val = np.array(predictions['val'])
y_pred_test = np.array(predictions['test'])
y_train = np.array(targets['train'])
y_val = np.array(targets['val'])
y_test = np.array(targets['test'])

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
# RESIDUAL ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("RESIDUAL ANALYSIS")
print("-" * 40)

residuals_test = y_test - y_pred_test

print(f"\nTest Set Residual Statistics:")
print(f"  Mean: {np.mean(residuals_test):.6f}")
print(f"  Std: {np.std(residuals_test):.6f}")

# Identify worst predictions
test_results_df = test_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
test_results_df['predicted'] = y_pred_test
test_results_df['residual'] = residuals_test
test_results_df['abs_error'] = np.abs(residuals_test)

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
test_results_df.to_csv('../results/industry_only_test_predictions.csv', index=False)
print("Test predictions saved to: ../results/industry_only_test_predictions.csv")

# Save train predictions
train_results_df = train_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
train_results_df['predicted'] = y_pred_train
train_results_df['residual'] = y_train - y_pred_train
train_results_df['abs_error'] = np.abs(train_results_df['residual'])
train_results_df.to_csv('../results/industry_only_train_predictions.csv', index=False)
print("Train predictions saved to: ../results/industry_only_train_predictions.csv")

# Save validation predictions
val_results_df = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
val_results_df['predicted'] = y_pred_val
val_results_df['residual'] = y_val - y_pred_val
val_results_df['abs_error'] = np.abs(val_results_df['residual'])
val_results_df.to_csv('../results/industry_only_val_predictions.csv', index=False)
print("Validation predictions saved to: ../results/industry_only_val_predictions.csv")

# Summary statistics
summary_stats = pd.DataFrame({
    'dataset': ['Train', 'Validation', 'Test'],
    'count': [len(y_train), len(y_val), len(y_test)],
    'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
    'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
    'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
    'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']]
})
summary_stats.to_csv('../results/industry_only_summary_stats.csv', index=False)
print("Summary statistics saved to: ../results/industry_only_summary_stats.csv")

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
axes[0, 0].set_title(f'Industry FFNN Only: Predicted vs Actual (R² = {test_metrics["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution
axes[0, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals_test), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(residuals_test):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Industry FFNN Only: Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Training History
axes[0, 2].plot(history['epoch'], history['train_loss'], label='Train', alpha=0.7)
axes[0, 2].plot(history['epoch'], history['val_loss'], label='Validation', alpha=0.7)
axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss (MSE)')
axes[0, 2].set_title('Industry FFNN Only: Training History')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Absolute Error Distribution
axes[1, 0].hist(np.abs(residuals_test), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=test_metrics['MAE'], color='r', linestyle='--', lw=2, 
                   label=f'MAE: {test_metrics["MAE"]:.4f}')
axes[1, 0].set_xlabel('Absolute Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Industry FFNN Only: Absolute Error Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Metrics Comparison
metrics_compare = pd.DataFrame({
    'Train': [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['R2']],
    'Validation': [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['R2']],
    'Test': [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['R2']]
}, index=['MAE', 'RMSE', 'R²'])

metrics_compare.T.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Industry FFNN Only: Metrics Across Datasets')
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
axes[1, 2].set_title('Industry FFNN Only: Log Scale Comparison')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/industry_only_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# INDUSTRY EMBEDDING ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("ANALYZING INDUSTRY EMBEDDINGS")
print("-" * 40)

# Get some example industries to visualize
example_industries = ['327310', '327320', '327330', '311111', '311210', '336111']
example_names = []
example_embeddings = []

with torch.no_grad():
    for ind_code in example_industries:
        if ind_code in industry_to_idx:
            idx = industry_to_idx[ind_code]
            emb = model.get_industry_embedding(torch.LongTensor([idx]).to(device)).cpu().numpy()[0]
            example_embeddings.append(emb)
            example_names.append(f"{ind_code}")

if example_embeddings:
    example_embeddings = np.array(example_embeddings)
    
    # Compute similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    ind_similarity = cosine_similarity(example_embeddings)
    
    # Plot similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(ind_similarity, xticklabels=example_names, yticklabels=example_names,
                annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Industry Embedding Similarities (Industry FFNN Only)')
    plt.tight_layout()
    plt.savefig('../results/plots/industry_only_similarities.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nIndustry similarities:")
    for i, name1 in enumerate(example_names):
        for j, name2 in enumerate(example_names):
            if i < j:
                print(f"  {name1} vs {name2}: {ind_similarity[i,j]:.4f}")

# ============================================================================
# SAVE ALL METRICS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING ALL METRICS")
print("-" * 40)

all_metrics = {
    'model_params': {
        'industry_embedding_dim': 32,
        'final_hidden_dim': 64,
        'dropout': 0.2,
        'total_params': total_params
    },
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'training_history': history,
    'best_epoch': best_epoch,
    'training_time': training_time
}

with open('../results/industry_only_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("All metrics saved to: ../results/industry_only_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("INDUSTRY FFNN ONLY - ABLATION STUDY - FINAL SUMMARY")
print("=" * 60)
print(f"Model Parameters: {total_params:,}")
print(f"Best epoch: {best_epoch}")
print(f"Training time: {training_time:.2f} seconds")
print("\n" + "-" * 40)
print("TEST SET PERFORMANCE")
print("-" * 40)
print(f"MAE:   {test_metrics['MAE']:.6f}")
print(f"RMSE:  {test_metrics['RMSE']:.6f}")
print(f"R²:    {test_metrics['R2']:.6f}")
print(f"MAPE:  {test_metrics['MAPE']:.2f}%")
print(f"SMAPE: {test_metrics['SMAPE']:.2f}%")
print("=" * 60)
print(f"Results saved to: ../results/")
print(f"Plots saved to: ../results/plots/")
print("=" * 60)