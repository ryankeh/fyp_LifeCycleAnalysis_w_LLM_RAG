"""
Script 2: Train neural network model with GPU support
Uses: train.csv, val.csv
Saves: best_model.pt, training_history.csv, model_info.json
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from pathlib import Path
import time
import joblib

# Create directories
Path("../models").mkdir(parents=True, exist_ok=True)
Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/plots").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("TRAINING SCRIPT")
print("=" * 60)
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA")
print("-" * 40)

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")

print(f"Train set: {len(train_df):,} rows")
print(f"Validation set: {len(val_df):,} rows")

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

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

# ============================================================================
# VERIFY NO NAN VALUES - WITH AUTO-FIX
# ============================================================================

print("\n" + "-" * 40)
print("VERIFYING NO NAN VALUES")
print("-" * 40)

train_nan = np.isnan(X_train).sum()
val_nan = np.isnan(X_val).sum()
train_inf = np.isinf(X_train).sum()
val_inf = np.isinf(X_val).sum()

print(f"X_train NaN count: {train_nan}")
print(f"X_train Inf count: {train_inf}")
print(f"X_val NaN count: {val_nan}")
print(f"X_val Inf count: {val_inf}")
print(f"y_train NaN count: {np.isnan(y_train).sum()}")
print(f"y_val NaN count: {np.isnan(y_val).sum()}")

# Auto-fix any remaining issues
if train_nan > 0 or val_nan > 0 or train_inf > 0 or val_inf > 0:
    print("\n⚠️ Found problematic values! Auto-fixing...")
    
    # Function to clean array
    def clean_array(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    
    X_train = clean_array(X_train)
    X_val = clean_array(X_val)
    
    print("\nAfter cleaning:")
    print(f"X_train NaN count: {np.isnan(X_train).sum()}")
    print(f"X_train Inf count: {np.isinf(X_train).sum()}")
    print(f"X_val NaN count: {np.isnan(X_val).sum()}")
    print(f"X_val Inf count: {np.isinf(X_val).sum()}")

# ============================================================================
# NORMALIZE FEATURES
# ============================================================================

print("\n" + "-" * 40)
print("NORMALIZING FEATURES")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')
print("Scaler saved to: ../models/scaler.pkl")

# Also scale target for better training
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
joblib.dump(target_scaler, '../models/target_scaler.pkl')
print("Target scaler saved to: ../models/target_scaler.pkl")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val_scaled).reshape(-1, 1).to(device)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = min(256, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ============================================================================
# DEFINE SIMPLE NEURAL NETWORK
# ============================================================================

print("\n" + "-" * 40)
print("DEFINING MODEL ARCHITECTURE")
print("-" * 40)

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

model = SimpleNN(input_dim=len(feature_cols)).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING SETUP")
print("-" * 40)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    # Training
    model.train()
    train_losses = []
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at epoch {epoch+1}, skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
    
    if len(train_losses) == 0:
        print(f"⚠️ No valid batches at epoch {epoch+1}, stopping")
        break
        
    avg_train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            if not torch.isnan(loss):
                val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
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
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_epoch = epoch + 1
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': best_val_loss,
            'feature_cols': feature_cols
        }, '../models/best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "-" * 40)
print("SAVING RESULTS")
print("-" * 40)

# Save history
history_df = pd.DataFrame(history)
history_df.to_csv('../results/training_history.csv', index=False)
print("Training history saved to: ../results/training_history.csv")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['epoch'], history['train_loss'], label='Train', alpha=0.7)
ax1.plot(history['epoch'], history['val_loss'], label='Validation', alpha=0.7)
if best_epoch > 0:
    ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['epoch'], history['lr'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model info
model_info = {
    'input_dim': len(feature_cols),
    'architecture': '64-32',
    'total_params': total_params,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'training_time': training_time,
    'num_epochs_trained': len(history['epoch'])
}

with open('../models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best model saved to: ../models/best_model.pt")
print(f"Best validation loss: {best_val_loss:.6f}")
print("=" * 60)