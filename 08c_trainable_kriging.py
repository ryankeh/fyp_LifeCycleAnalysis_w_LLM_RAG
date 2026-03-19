"""
Script B2: Trainable Kriging Models (per industry)
For each industry, learns spatial correlation from distances and neighbor carbon intensities
Outputs: trained variogram parameters, kriging predictions with uncertainty (with composite keys)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path
from tqdm import tqdm

# Create directories
Path("../results/bayesian").mkdir(parents=True, exist_ok=True)
Path("../models/bayesian/kriging").mkdir(parents=True, exist_ok=True)
Path("../results/plots/bayesian/kriging").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("TRAINABLE KRIGING MODELS (PER INDUSTRY)")
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
    
    print(f"\n{dataset_name} Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return metrics

# ============================================================================
# DIFFERENTIABLE KRIGING MODULE
# ============================================================================

class VariogramModel(nn.Module):
    """Exponential variogram with trainable parameters"""
    def __init__(self, nugget_init=0.1, sill_init=1.0, range_init=1000.0):
        super().__init__()
        self.log_nugget = nn.Parameter(torch.log(torch.tensor(nugget_init)))
        self.log_sill = nn.Parameter(torch.log(torch.tensor(sill_init)))
        self.log_range = nn.Parameter(torch.log(torch.tensor(range_init)))
        
    def forward(self, h):
        """Compute variogram value at distance h"""
        nugget = torch.exp(self.log_nugget)
        sill = torch.exp(self.log_sill)
        rng = torch.exp(self.log_range)
        
        # Exponential variogram: γ(h) = nugget + sill * (1 - exp(-h/rng))
        gamma = nugget + sill * (1 - torch.exp(-h / rng))
        return gamma
    
    def covariance(self, h):
        """Covariance C(h) = sill + nugget - γ(h)"""
        nugget = torch.exp(self.log_nugget)
        sill = torch.exp(self.log_sill)
        rng = torch.exp(self.log_range)
        
        cov = sill * torch.exp(-h / rng)
        cov = torch.where(h == 0, sill + nugget, cov)
        return cov

class DifferentiableKriging:
    """Performs kriging with differentiable parameters"""
    
    def __init__(self, variogram_model):
        self.variogram = variogram_model
        self.eps = 1e-6
        
    def krige(self, target_distances, target_values, distances_matrix):
        """
        Perform kriging prediction
        
        Args:
            target_distances: distances from target to each known point [n_neighbors]
            target_values: values at known points [n_neighbors]
            distances_matrix: distances between known points [n_neighbors, n_neighbors]
        
        Returns:
            prediction, variance
        """
        n = len(target_values)
        
        # Build kriging matrices
        Gamma = self.variogram.covariance(distances_matrix)  # [n, n]
        gamma = self.variogram.covariance(target_distances)  # [n]
        
        # Add small diagonal for numerical stability
        Gamma = Gamma + self.eps * torch.eye(n, device=Gamma.device)
        
        # Solve for weights: Γw = γ
        try:
            weights = torch.linalg.solve(Gamma, gamma)
        except:
            # Fallback to pseudo-inverse
            weights = torch.linalg.lstsq(Gamma, gamma.unsqueeze(1))[0].squeeze()
        
        # Prediction
        prediction = torch.dot(weights, target_values)
        
        # Kriging variance
        variance = self.variogram.covariance(torch.tensor(0.0)) - torch.dot(weights, gamma)
        variance = torch.clamp(variance, min=self.eps)
        
        return prediction, variance

# ============================================================================
# LOAD DATA WITH INDICES
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA")
print("-" * 40)

# Load datasets and add original indices
train_df = pd.read_csv("../data/train.csv").reset_index().rename(columns={'index': 'orig_index'})
val_df = pd.read_csv("../data/val.csv").reset_index().rename(columns={'index': 'orig_index'})
test_df = pd.read_csv("../data/test.csv").reset_index().rename(columns={'index': 'orig_index'})

print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

# Load distance matrix
dist_df = pd.read_csv("country_distance_matrix_distwces.csv", index_col=0)

# Get country mappings
countries = sorted(train_df['Country Code'].unique())
industries = sorted(train_df['industry_code'].unique())

country_to_idx = {c: i for i, c in enumerate(countries)}
idx_to_country = {i: c for i, c in enumerate(countries)}

n_countries = len(countries)
n_industries = len(industries)

print(f"Countries: {n_countries}, Industries: {n_industries}")

# ============================================================================
# TRAIN KRIGING MODELS PER INDUSTRY
# ============================================================================

print("\n" + "-" * 40)
print("TRAINING KRIGING MODELS (PER INDUSTRY)")
print("-" * 40)

# Store results with ALL needed information
industry_params = {}

# For each split, store: predictions, stds, targets, AND the identifying information
kriging_results_data = {
    'train': {'orig_index': [], 'country': [], 'industry': [], 'true': [], 'pred': [], 'std': []},
    'val': {'orig_index': [], 'country': [], 'industry': [], 'true': [], 'pred': [], 'std': []},
    'test': {'orig_index': [], 'country': [], 'industry': [], 'true': [], 'pred': [], 'std': []}
}

# Process each industry
for industry_idx, industry in enumerate(tqdm(industries, desc="Training industries")):
    
    # Get data for this industry
    industry_train = train_df[train_df['industry_code'] == industry]
    industry_val = val_df[val_df['industry_code'] == industry]
    industry_test = test_df[test_df['industry_code'] == industry]
    
    if len(industry_train) < 5:  # Skip if too few samples
        continue
    
    # Prepare training data for this industry
    train_countries = industry_train['Country Code'].values
    train_values = industry_train['carbon_intensity'].values.astype(np.float32)
    
    # Get indices and distances
    train_country_indices = [country_to_idx[c] for c in train_countries]
    
    # Build distance matrix for training points
    n_train = len(train_country_indices)
    dist_matrix_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            c1 = idx_to_country[train_country_indices[i]]
            c2 = idx_to_country[train_country_indices[j]]
            dist_matrix_train[i, j] = dist_df.loc[c1, c2]
    
    # Convert to tensors
    train_values_tensor = torch.FloatTensor(train_values).to(device)
    dist_matrix_train_tensor = torch.FloatTensor(dist_matrix_train).to(device)
    
    # Initialize variogram model
    variogram = VariogramModel().to(device)
    optimizer = optim.Adam(variogram.parameters(), lr=0.01)
    
    # Train variogram parameters
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Compute empirical variogram
        diffs = train_values_tensor.unsqueeze(1) - train_values_tensor.unsqueeze(0)
        squared_diffs = diffs ** 2
        gamma_empirical = 0.5 * squared_diffs
        
        # Model prediction
        gamma_model = variogram(dist_matrix_train_tensor)
        
        # Loss: MSE between empirical and modeled variogram
        loss = torch.mean((gamma_empirical - gamma_model) ** 2)
        
        loss.backward()
        optimizer.step()
    
    # Store trained parameters
    industry_params[industry] = {
        'nugget': torch.exp(variogram.log_nugget).item(),
        'sill': torch.exp(variogram.log_sill).item(),
        'range': torch.exp(variogram.log_range).item()
    }
    
    # Create kriging object
    kriging = DifferentiableKriging(variogram)
    
    # Predict for each split
    for split_name, split_df in [('train', industry_train), ('val', industry_val), ('test', industry_test)]:
        
        for _, row in split_df.iterrows():
            # Store ALL identifying information
            orig_idx = row['orig_index']
            target_country = row['Country Code']
            target_industry = row['industry_code']
            true_value = row['carbon_intensity']
            
            # Find all OTHER training points (for kriging)
            other_mask = industry_train['Country Code'] != target_country
            other_countries = industry_train.loc[other_mask, 'Country Code'].values
            other_values = industry_train.loc[other_mask, 'carbon_intensity'].values.astype(np.float32)
            
            if len(other_values) < 3:  # Not enough neighbors
                if split_name == 'train':
                    pred_value = true_value
                    pred_std = 0.1
                else:
                    # Use training mean as fallback
                    pred_value = np.mean(train_values)
                    pred_std = np.std(train_values)
                
                # Store results with all identifying info
                kriging_results_data[split_name]['orig_index'].append(orig_idx)
                kriging_results_data[split_name]['country'].append(target_country)
                kriging_results_data[split_name]['industry'].append(target_industry)
                kriging_results_data[split_name]['true'].append(true_value)
                kriging_results_data[split_name]['pred'].append(pred_value)
                kriging_results_data[split_name]['std'].append(pred_std)
                continue
            
            # Get distances
            # Distances from target to others
            target_dists = []
            for oc in other_countries:
                target_dists.append(dist_df.loc[target_country, oc])
            
            # Distance matrix among others
            n_other = len(other_countries)
            other_dists = np.zeros((n_other, n_other))
            for i in range(n_other):
                for j in range(n_other):
                    c1 = other_countries[i]
                    c2 = other_countries[j]
                    other_dists[i, j] = dist_df.loc[c1, c2]
            
            # Convert to tensors
            target_dists_tensor = torch.FloatTensor(target_dists).to(device)
            other_values_tensor = torch.FloatTensor(other_values).to(device)
            other_dists_tensor = torch.FloatTensor(other_dists).to(device)
            
            # Kriging prediction
            with torch.no_grad():
                pred, var = kriging.krige(target_dists_tensor, other_values_tensor, other_dists_tensor)
                
                # Store results with all identifying info
                kriging_results_data[split_name]['orig_index'].append(orig_idx)
                kriging_results_data[split_name]['country'].append(target_country)
                kriging_results_data[split_name]['industry'].append(target_industry)
                kriging_results_data[split_name]['true'].append(true_value)
                kriging_results_data[split_name]['pred'].append(pred.item())
                kriging_results_data[split_name]['std'].append(np.sqrt(var.item()))

# ============================================================================
# EVALUATE AND SAVE KRIGING MODELS
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATING KRIGING MODELS")
print("-" * 40)

kriging_results = {}

for split in ['train', 'val', 'test']:
    if len(kriging_results_data[split]['pred']) > 0:
        # Convert to arrays
        y_true = np.array(kriging_results_data[split]['true'])
        y_pred = np.array(kriging_results_data[split]['pred'])
        y_std = np.array(kriging_results_data[split]['std'])
        
        # Calculate metrics
        metrics = evaluate_predictions(y_true, y_pred, y_std, f"KRIGING {split.upper()}")
        kriging_results[split] = metrics
        
        # Create DataFrame with ALL identifying information
        split_df = pd.DataFrame({
            'Country Code': kriging_results_data[split]['country'],
            'industry_code': kriging_results_data[split]['industry'],
            'orig_index': kriging_results_data[split]['orig_index'],
            'true': y_true,
            'predicted': y_pred,
            'std': y_std
        })
        
        # Save to CSV
        output_file = f'../results/bayesian/kriging_{split}_predictions.csv'
        split_df.to_csv(output_file, index=False)
        print(f"Saved {len(split_df)} predictions for {split} set to {output_file}")
        
        # Verify the saved file has the right columns
        print(f"  Columns: {split_df.columns.tolist()}")
        print(f"  Sample: Country={split_df['Country Code'].iloc[0]}, Industry={split_df['industry_code'].iloc[0]}")

# ============================================================================
# SAVE MODELS AND RESULTS
# ============================================================================

# Save industry parameters
params_df = pd.DataFrame(industry_params).T
params_df.to_csv('../results/bayesian/kriging_parameters.csv')
print(f"\nKriging parameters saved for {len(industry_params)} industries")

# Save overall metrics
with open('../results/bayesian/kriging_metrics.json', 'w') as f:
    json.dump(kriging_results, f, indent=2, default=str)

# Plot parameter distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

params_df = pd.DataFrame(industry_params).T
for ax, param in zip(axes, ['nugget', 'sill', 'range']):
    ax.hist(params_df[param], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel(param)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {param}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/bayesian/kriging_parameters.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("KRIGING TRAINING COMPLETE")
print(f"Models trained for {len(industry_params)} industries")
print("=" * 60)