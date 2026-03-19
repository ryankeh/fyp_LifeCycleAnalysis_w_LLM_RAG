"""
Script B1: Bayesian Neural Network for Industry Variables
Trains a Bayesian FFNN on the 8 industry variables
Outputs: trained model, predictions with uncertainty, metrics (all splits)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchbnn as bnn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path

# Create directories
Path("../results/bayesian").mkdir(parents=True, exist_ok=True)
Path("../models/bayesian").mkdir(parents=True, exist_ok=True)
Path("../results/plots/bayesian").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("BAYESIAN NEURAL NETWORK - INDUSTRY MODEL")
print("=" * 60)
print(f"Using device: {device}")

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true, y_pred, y_std=None, dataset_name="Dataset"):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'MedAE': np.median(np.abs(y_true - y_pred))
    }
    
    if y_std is not None:
        coverage_68 = np.mean((y_true >= y_pred - y_std) & (y_true <= y_pred + y_std))
        coverage_95 = np.mean((y_true >= y_pred - 2*y_std) & (y_true <= y_pred + 2*y_std))
        metrics['Coverage_68'] = coverage_68
        metrics['Coverage_95'] = coverage_95
        metrics['Avg_Std'] = np.mean(y_std)
    
    print(f"\n{dataset_name} Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return metrics

# ============================================================================
# BAYESIAN NEURAL NETWORK MODEL
# ============================================================================

class BayesianIndustryNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[32, 16]):
        super(BayesianIndustryNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, 
                                         in_features=prev_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.mean_layer = nn.Linear(prev_dim, 1)
        self.logvar_layer = nn.Linear(prev_dim, 1)
        
        self.hidden_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        features = self.hidden_layers(x)
        mean = self.mean_layer(features).squeeze()
        logvar = self.logvar_layer(features).squeeze()
        return mean, logvar.exp()

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA")
print("-" * 40)

# Load datasets and add split identifiers
train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

# Add split column for later identification
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

industry_vars = ['process_emission_intensity_score', 'material_processing_depth_score',
                 'thermal_process_intensity_score', 'electrification_feasibility_score',
                 'continuous_operations_intensity_score', 'material_throughput_scale_score',
                 'chemical_intensity_score', 'capital_vs_labor_intensity_score']

# Prepare data
X_train = train_df[industry_vars].values.astype(np.float32)
y_train = train_df['carbon_intensity'].values.astype(np.float32)
X_val = val_df[industry_vars].values.astype(np.float32)
y_val = val_df['carbon_intensity'].values.astype(np.float32)
X_test = test_df[industry_vars].values.astype(np.float32)
y_test = test_df['carbon_intensity'].values.astype(np.float32)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scale target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

# Data loaders
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING BAYESIAN NN")
print("-" * 40)

model = BayesianIndustryNet(input_dim=8, hidden_dims=[32, 16]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

num_epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0
history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'kl_loss': []}

print("Training loop:")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_kl = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        mean, var = model(batch_X)
        
        nll = 0.5 * torch.log(var) + 0.5 * ((batch_y - mean)**2) / var
        nll = nll.mean()
        kl = kl_loss(model)
        loss = nll + kl / len(train_loader)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_kl += kl.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_kl = train_kl / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        mean_val, var_val = model(X_val_tensor)
        val_nll = 0.5 * torch.log(var_val) + 0.5 * ((y_val_tensor - mean_val)**2) / var_val
        val_loss = val_nll.mean().item()
    
    scheduler.step(val_loss)
    
    history['epoch'].append(epoch+1)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['kl_loss'].append(avg_kl)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}, KL={avg_kl:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '../models/bayesian/industry_bnn.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('../models/bayesian/industry_bnn.pt'))

# ============================================================================
# EVALUATION AND SAVING
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATION AND SAVING PREDICTIONS")
print("-" * 40)

model.eval()
with torch.no_grad():
    mean_train, var_train = model(X_train_tensor)
    mean_val, var_val = model(X_val_tensor)
    mean_test, var_test = model(X_test_tensor)
    
    y_pred_train = scaler_y.inverse_transform(mean_train.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_val = scaler_y.inverse_transform(mean_val.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_test = scaler_y.inverse_transform(mean_test.cpu().numpy().reshape(-1, 1)).flatten()
    
    std_train = np.sqrt(var_train.cpu().numpy()) * scaler_y.scale_
    std_val = np.sqrt(var_val.cpu().numpy()) * scaler_y.scale_
    std_test = np.sqrt(var_test.cpu().numpy()) * scaler_y.scale_

# Evaluate
train_metrics = evaluate_predictions(y_train, y_pred_train, std_train, "TRAIN")
val_metrics = evaluate_predictions(y_val, y_pred_val, std_val, "VALIDATION")
test_metrics = evaluate_predictions(y_test, y_pred_test, std_test, "TEST")

# Save scalers
joblib.dump(scaler_X, '../models/bayesian/industry_scaler_X.pkl')
joblib.dump(scaler_y, '../models/bayesian/industry_scaler_y.pkl')

# Save predictions for ALL splits (using Country Code and industry_code as keys)
for split_name, split_df, predictions, stds in [
    ('train', train_df, y_pred_train, std_train),
    ('val', val_df, y_pred_val, std_val),
    ('test', test_df, y_pred_test, std_test)
]:
    results_df = pd.DataFrame({
        'Country Code': split_df['Country Code'].values,
        'industry_code': split_df['industry_code'].values,
        'Country': split_df['Country'].values,
        'industry_name': split_df['industry_name'].values,
        'carbon_intensity': split_df['carbon_intensity'].values,
        'industry_pred': predictions,
        'industry_std': stds,
        'split': split_name
    })
    results_df.to_csv(f'../results/bayesian/industry_predictions_{split_name}.csv', index=False)
    print(f"Saved {len(results_df)} predictions for {split_name} set")

# Save metrics
all_metrics = {
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'history': history
}
with open('../results/bayesian/industry_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['epoch'], history['train_loss'], label='Train')
axes[0].plot(history['epoch'], history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History - Industry BNN')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['epoch'], history['kl_loss'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('KL Loss')
axes[1].set_title('KL Divergence - Industry BNN')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/bayesian/industry_training.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("INDUSTRY BNN COMPLETE")
print(f"Test R²: {test_metrics['R2']:.4f}")
print("=" * 60)