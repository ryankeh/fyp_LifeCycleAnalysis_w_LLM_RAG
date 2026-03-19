"""
Script B4: Bayesian Meta-Learner (3-Model Ensemble)
Combines industry BNN, country BNN, and kriging predictions with uncertainty
Outputs: final ensemble predictions with uncertainty, full metrics
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
Path("../results/bayesian/meta").mkdir(parents=True, exist_ok=True)
Path("../models/bayesian/meta").mkdir(parents=True, exist_ok=True)
Path("../results/plots/bayesian/meta").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("BAYESIAN META-LEARNER (3-MODEL ENSEMBLE)")
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
# NUMERICALLY STABLE BAYESIAN META-NETWORK
# ============================================================================

class StableBayesianMetaNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[24, 12]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Use smaller prior sigma for stability
            layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, 
                                         in_features=prev_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Separate heads for mean and log variance
        self.hidden_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, 1)
        self.logvar_layer = nn.Linear(prev_dim, 1)
        
        # Initialize logvar layer to predict small values
        nn.init.zeros_(self.logvar_layer.weight)
        nn.init.constant_(self.logvar_layer.bias, -2.0)  # Start with var = exp(-2) ≈ 0.135
        
    def forward(self, x):
        features = self.hidden_layers(x)
        
        # Mean prediction (unconstrained)
        mean = self.mean_layer(features).squeeze()
        
        # Log variance with constraints
        logvar = self.logvar_layer(features).squeeze()
        
        # ===== NUMERICAL STABILITY FIXES =====
        # 1. Clamp logvar to reasonable range
        logvar = torch.clamp(logvar, min=-5, max=5)  # var in [0.0067, 148.4]
        
        # 2. Compute variance safely
        var = torch.exp(logvar)
        
        # 3. Add small epsilon for absolute safety
        var = var + 1e-6
        
        return mean, var

# Replace your model initialization with:
model = StableBayesianMetaNet(input_dim=6, hidden_dims=[24, 12]).to(device)

# ============================================================================
# LOAD META DATA
# ============================================================================

print("\n" + "-" * 40)
print("LOADING META-LEARNER DATA")
print("-" * 40)

# Load pre-split data from 08d_prepare_meta_input.py
try:
    X_train = np.load('../results/bayesian/meta/X_train.npy')
    y_train = np.load('../results/bayesian/meta/y_train.npy')
    X_val = np.load('../results/bayesian/meta/X_val.npy')
    y_val = np.load('../results/bayesian/meta/y_val.npy')
    X_test = np.load('../results/bayesian/meta/X_test.npy')
    y_test = np.load('../results/bayesian/meta/y_test.npy')
    
    print(f"Loaded pre-split data:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
except FileNotFoundError:
    print("Pre-split data not found. Loading combined data and splitting...")
    # Fallback to loading combined data
    X_meta = np.load('../results/bayesian/meta/X_meta.npy')
    y_meta = np.load('../results/bayesian/meta/y_meta.npy')
    
    print(f"X shape: {X_meta.shape}")
    print(f"y shape: {y_meta.shape}")
    
    # Train/val/test split
    from sklearn.model_selection import train_test_split
    
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_meta, y_meta, test_size=0.2, random_state=42
    )
    
    # Second split: 50% of temp for val, 50% for test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

# Load feature names
with open('../results/bayesian/meta/feature_names.json', 'r') as f:
    feature_names = json.load(f)

print(f"\nFeatures (6 total):")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")

print(f"\nFinal split sizes:")
print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

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

# Data loader
batch_size = min(64, len(X_train))
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ============================================================================
# DIAGNOSTIC: Check data statistics
# ============================================================================

print("\n" + "-" * 40)
print("DATA DIAGNOSTICS")
print("-" * 40)

print(f"y_train stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}, min={y_train.min():.4f}, max={y_train.max():.4f}")
print(f"y_val stats: mean={y_val.mean():.4f}, std={y_val.std():.4f}")
print(f"y_test stats: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

# Check if there's any variation in the data
if y_train.std() < 1e-6:
    print("WARNING: Target variable has near-zero variance!")

# Check feature correlations with target
print("\nFeature correlations with target:")
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
    print(f"  {name}: {corr:.4f}")

# ============================================================================
# NUMERICALLY STABLE TRAINING LOOP (FIXED)
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING STABLE BAYESIAN META-LEARNER")
print("-" * 40)

model = StableBayesianMetaNet(input_dim=6, hidden_dims=[24, 12]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

num_epochs = 100
best_val_loss = float('inf')
patience = 15
patience_counter = 0
history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'kl_loss': []}

# Gradient clipping value
max_grad_norm = 0.5

# Create validation loader once
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Training loop:")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_kl = 0
    train_nll = 0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with safety
        try:
            mean, var = model(batch_X)
        except Exception as e:
            print(f"  Forward pass failed: {e}")
            continue
        
        # ===== COMPREHENSIVE NaN CHECK =====
        has_nan = False
        for name, tensor in [('mean', mean), ('var', var)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"  NaN/Inf detected in {name}")
                has_nan = True
                break
        
        if has_nan:
            continue
        
        # ===== STABLE NLL COMPUTATION =====
        # Ensure var is absolutely safe
        var_safe = torch.clamp(var, min=1e-4, max=10.0)
        
        # Compute error
        error = batch_y - mean
        
        # Log variance term
        log_var = torch.log(var_safe)
        
        # Squared error term with safety
        squared_error = error ** 2
        nll = 0.5 * log_var + 0.5 * squared_error / var_safe
        nll = nll.mean()
        
        # Check NLL
        if torch.isnan(nll) or torch.isinf(nll):
            print(f"  NLL is NaN/Inf")
            continue
        
        # KL divergence (with safety)
        kl = kl_loss(model)
        if torch.isnan(kl) or torch.isinf(kl):
            print(f"  KL is NaN/Inf")
            continue
        
        # Annealed KL weight
        beta = min(1.0, epoch / 20) / len(train_loader)
        loss = nll + beta * kl
        
        # Check total loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Total loss is NaN/Inf")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Check for NaN gradients
        grad_has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  NaN gradient in {name}")
                    grad_has_nan = True
                    break
        
        if grad_has_nan:
            optimizer.zero_grad()  # Reset gradients
            continue
        
        optimizer.step()
        
        train_loss += loss.item()
        train_kl += kl.item()
        train_nll += nll.item()
        batch_count += 1
    
    if batch_count == 0:
        print(f"Warning: No valid batches in epoch {epoch+1}")
        # Reset optimizer state if all batches failed
        optimizer.zero_grad()
        continue
    
    # Averages
    avg_train_loss = train_loss / batch_count
    avg_kl = train_kl / batch_count
    avg_nll = train_nll / batch_count
    
    # Validation phase
    model.eval()
    val_nll = 0
    val_count = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:  # Now val_loader is defined
            try:
                mean, var = model(batch_X)
                
                # Same safety checks as training
                var_safe = torch.clamp(var, min=1e-4, max=10.0)
                error = batch_y - mean
                log_var = torch.log(var_safe)
                squared_error = error ** 2
                nll = 0.5 * log_var + 0.5 * squared_error / var_safe
                nll = nll.mean()
                
                if not torch.isnan(nll) and not torch.isinf(nll):
                    val_nll += nll.item()
                    val_count += 1
            except Exception as e:
                continue
    
    avg_val_nll = val_nll / val_count if val_count > 0 else float('inf')
    
    # Update scheduler only with valid loss
    if avg_val_nll != float('inf'):
        scheduler.step(avg_val_nll)
    
    # Store history
    history['epoch'].append(epoch+1)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_nll if avg_val_nll != float('inf') else np.nan)
    history['kl_loss'].append(avg_kl)
    
    # Print progress
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.6f} (NLL={avg_nll:.4f}, KL={avg_kl:.4f}), Val Loss={avg_val_nll if avg_val_nll!=float('inf') else 0:.6f}")
    
    # Early stopping
    if avg_val_nll < best_val_loss:
        best_val_loss = avg_val_nll
        patience_counter = 0
        torch.save(model.state_dict(), '../models/bayesian/meta/meta_bnn_stable.pt')
        print(f"  ✓ Saved best model (val_loss={best_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATING META-LEARNER")
print("-" * 40)

model.eval()
with torch.no_grad():
    mean_train, var_train = model(X_train_tensor)
    mean_val, var_val = model(X_val_tensor)
    mean_test, var_test = model(X_test_tensor)
    
    # Convert back
    y_pred_train = scaler_y.inverse_transform(mean_train.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_val = scaler_y.inverse_transform(mean_val.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_test = scaler_y.inverse_transform(mean_test.cpu().numpy().reshape(-1, 1)).flatten()
    
    std_train = np.sqrt(var_train.cpu().numpy()) * scaler_y.scale_
    std_val = np.sqrt(var_val.cpu().numpy()) * scaler_y.scale_
    std_test = np.sqrt(var_test.cpu().numpy()) * scaler_y.scale_

# Evaluate
train_metrics = evaluate_predictions(y_train, y_pred_train, std_train, "META TRAIN")
val_metrics = evaluate_predictions(y_val, y_pred_val, std_val, "META VALIDATION")
test_metrics = evaluate_predictions(y_test, y_pred_test, std_test, "META TEST")

# ============================================================================
# COMPARE WITH INDIVIDUAL MODELS
# ============================================================================

print("\n" + "-" * 40)
print("MODEL COMPARISON (3-MODEL ENSEMBLE)")
print("-" * 40)

# Compute simple average
simple_avg = (X_test[:, 0] + X_test[:, 2] + X_test[:, 4]) / 3

# Compute uncertainty-weighted average for test set
inv_var_ind = 1 / (X_test[:, 1]**2 + 1e-8)  # industry_std^2
inv_var_country = 1 / (X_test[:, 3]**2 + 1e-8)  # country_std^2
inv_var_krige = 1 / (X_test[:, 5]**2 + 1e-8)  # krige_std^2
total_inv_var = inv_var_ind + inv_var_country + inv_var_krige

weighted_pred = (inv_var_ind * X_test[:, 0] + 
                 inv_var_country * X_test[:, 2] + 
                 inv_var_krige * X_test[:, 4]) / total_inv_var

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Industry BNN', 'Country BNN', 'Kriging', 
              'Simple Avg (3)', 'Uncertainty-Weighted', 'Bayesian Meta'],
    'MAE': [
        mean_absolute_error(y_test, X_test[:, 0]),
        mean_absolute_error(y_test, X_test[:, 2]),
        mean_absolute_error(y_test, X_test[:, 4]),
        mean_absolute_error(y_test, simple_avg),
        mean_absolute_error(y_test, weighted_pred),
        test_metrics['MAE']
    ],
    'RMSE': [
        rmse(y_test, X_test[:, 0]),
        rmse(y_test, X_test[:, 2]),
        rmse(y_test, X_test[:, 4]),
        rmse(y_test, simple_avg),
        rmse(y_test, weighted_pred),
        test_metrics['RMSE']
    ],
    'R2': [
        r2_score(y_test, X_test[:, 0]),
        r2_score(y_test, X_test[:, 2]),
        r2_score(y_test, X_test[:, 4]),
        r2_score(y_test, simple_avg),
        r2_score(y_test, weighted_pred),
        test_metrics['R2']
    ]
})

print("\nComparison Table:")
print(comparison.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING VISUALIZATIONS")
print("-" * 40)

# After training, plot the history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['epoch'], history['train_loss'], label='Train', linewidth=2)
axes[0].plot(history['epoch'], history['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['epoch'], history['kl_loss'], color='green', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('KL Loss')
axes[1].set_title('KL Divergence')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/bayesian/meta/training_curve.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title(f'Meta-Learner: Predicted vs Actual (R²={test_metrics["R2"]:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals
residuals = y_test - y_pred_test
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].grid(True, alpha=0.3)

# 3. Training history
axes[0, 2].plot(history['epoch'], history['train_loss'], label='Train')
axes[0, 2].plot(history['epoch'], history['val_loss'], label='Validation')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('Training History')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Uncertainty calibration
axes[1, 0].errorbar(y_test[:200], y_pred_test[:200], yerr=2*std_test[:200], 
                    fmt='o', alpha=0.3, capsize=2)
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Predicted ± 2σ')
axes[1, 0].set_title('Predictions with Uncertainty (95% CI)')
axes[1, 0].grid(True, alpha=0.3)

# 5. Model comparison bar chart
x = np.arange(len(comparison))
width = 0.25
axes[1, 1].bar(x - width, comparison['MAE'], width, label='MAE', alpha=0.8)
axes[1, 1].bar(x, comparison['RMSE'], width, label='RMSE', alpha=0.8)
# Scale R² to be on similar scale for visualization
r2_scaled = (comparison['R2'] + 1) / 2  # Map [-1,1] to [0,1]
axes[1, 1].bar(x + width, r2_scaled, width, label='R² (scaled)', alpha=0.8)
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Model Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(comparison['Model'], rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Feature importance
# Extract mean weights from first Bayesian layer
weights = []
with torch.no_grad():
    for layer in model.hidden_layers:
        if isinstance(layer, bnn.BayesLinear):
            weights.append(layer.weight_mu.cpu().numpy())

if weights:
    # Average absolute weights across first hidden units
    first_layer_weights = np.abs(weights[0]).mean(axis=0)
    axes[1, 2].barh(feature_names, first_layer_weights)
    axes[1, 2].set_xlabel('Avg |Weight|')
    axes[1, 2].set_title('Meta-Learner Feature Importance')
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/bayesian/meta/meta_results_3model.png', dpi=150)
plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING RESULTS")
print("-" * 40)

# Save predictions
results_df = pd.DataFrame({
    'true': y_test,
    'predicted': y_pred_test,
    'std': std_test,
    'industry_pred': X_test[:, 0],
    'industry_std': X_test[:, 1],
    'country_pred': X_test[:, 2],
    'country_std': X_test[:, 3],
    'krige_pred': X_test[:, 4],
    'krige_std': X_test[:, 5]
})
results_df.to_csv('../results/bayesian/meta/final_predictions_3model.csv', index=False)

# Save metrics
all_metrics = {
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'comparison': comparison.to_dict('records'),
    'history': history,
    'feature_names': feature_names
}
with open('../results/bayesian/meta/meta_metrics_3model.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Save scalers
joblib.dump(scaler_X, '../models/bayesian/meta/meta_scaler_X_3model.pkl')
joblib.dump(scaler_y, '../models/bayesian/meta/meta_scaler_y_3model.pkl')

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': 6,
    'hidden_dims': [24, 12],
    'feature_names': feature_names
}, '../models/bayesian/meta/meta_bnn_complete_3model.pt')

print("\n" + "=" * 60)
print("BAYESIAN 3-MODEL ENSEMBLE COMPLETE")
print(f"Final Test R²: {test_metrics['R2']:.4f}")
print("=" * 60)

# Print ensemble weights learned by meta-learner
print("\nLearned Ensemble Strategy:")
print("-" * 40)
if weights:
    print("\nFeature importance scores (higher = more important):")
    for name, weight in zip(feature_names, first_layer_weights):
        print(f"  {name}: {weight:.4f}")