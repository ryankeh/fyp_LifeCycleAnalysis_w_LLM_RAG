"""
Script: Hyperparameter Tuning for GCN Distance Scale
Tests different scale values: 5000, 1000, 500, 200, 100, 50
Standalone version - uses classes from 07_gcn.py but doesn't import
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
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results/tuning").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print("GCN DISTANCE SCALE HYPERPARAMETER TUNING")
print("=" * 80)
print(f"Using device: {device}")

# ============================================================================
# GCN LAYER IMPLEMENTATION (copied from 07_gcn.py)
# ============================================================================

class GraphConvLayer(nn.Module):
    """Simple Graph Convolution layer"""
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        support = torch.mm(adj, x)
        out = self.linear(support)
        return out

class GCN(nn.Module):
    """2-layer Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, dropout=0.2):
        super(GCN, self).__init__()
        
        self.gcn1 = GraphConvLayer(input_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x

class IndustryEncoder(nn.Module):
    """Creates industry embeddings from the 8 industry variables"""
    def __init__(self, num_industries, industry_variables, embedding_dim=32):
        super(IndustryEncoder, self).__init__()
        
        self.register_buffer('industry_vars', 
                           torch.FloatTensor(industry_variables))
        
        self.projection = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim)
        )
        
        self.residual = nn.Embedding(num_industries, embedding_dim)
        nn.init.zeros_(self.residual.weight)
        
    def forward(self, industry_idx):
        vars_batch = self.industry_vars[industry_idx]
        projected = self.projection(vars_batch)
        residual = self.residual(industry_idx)
        industry_embeddings = projected + 0.1 * residual
        return industry_embeddings

class CountryIndustryPredictor(nn.Module):
    """Complete model: GCN for countries + IndustryEncoder for industries"""
    def __init__(self, num_countries, num_industries, industry_variables,
                 country_input_dim, country_hidden_dim=64, country_output_dim=32,
                 industry_embedding_dim=32, final_hidden_dim=64, dropout=0.2):
        super(CountryIndustryPredictor, self).__init__()
        
        self.gcn = GCN(
            input_dim=country_input_dim,
            hidden_dim=country_hidden_dim,
            output_dim=country_output_dim,
            dropout=dropout
        )
        
        self.industry_encoder = IndustryEncoder(
            num_industries=num_industries,
            industry_variables=industry_variables,
            embedding_dim=industry_embedding_dim
        )
        
        self.country_features = None
        self.adjacency = None
        self.country_embeddings = None
        
        self.predictor = nn.Sequential(
            nn.Linear(country_output_dim + industry_embedding_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim // 2, 1)
        )
        
    def set_graph_data(self, country_features, adjacency):
        self.country_features = country_features
        self.adjacency = adjacency
        with torch.no_grad():
            self.country_embeddings = self.gcn(country_features, adjacency)
        
    def forward(self, country_idx, industry_idx):
        country_emb = self.country_embeddings[country_idx]
        industry_emb = self.industry_encoder(industry_idx)
        combined = torch.cat([country_emb, industry_emb], dim=1)
        output = self.predictor(combined)
        return output.squeeze()

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_predictions(y_true, y_pred, dataset_name="Dataset"):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'MedAE': np.median(np.abs(y_true - y_pred))
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    
    return metrics

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLOADING DATA...")

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

with open('../data/feature_info.json', 'r') as f:
    feature_info = json.load(f)

feature_cols = feature_info['feature_names']
target_col = feature_info['target_name']

# Country features
country_features_cols = [col for col in feature_cols if col in 
                         ['gdp_per_capita_ppp', 'industry_value_added_pct', 
                          'renewable_energy_pct', 'coal_electricity_pct',
                          'energy_intensity_level', 'gdp_per_energy_unit',
                          'urban_population_pct', 'natural_resources_rents_pct']]

# Industry variables
industry_variables_cols = [col for col in feature_cols if col in
                          ['process_emission_intensity_score', 'material_processing_depth_score',
                           'thermal_process_intensity_score', 'electrification_feasibility_score',
                           'continuous_operations_intensity_score', 'material_throughput_scale_score',
                           'chemical_intensity_score', 'capital_vs_labor_intensity_score']]

# Get unique countries and industries
countries = sorted(train_df['Country Code'].unique())
industries = sorted(train_df['industry_code'].unique())

country_to_idx = {country: i for i, country in enumerate(countries)}
industry_to_idx = {industry: i for i, industry in enumerate(industries)}

num_countries = len(countries)
num_industries = len(industries)

print(f"Countries: {num_countries} | Industries: {num_industries}")

# ============================================================================
# LOAD DISTANCE MATRIX
# ============================================================================

print("\nLOADING DISTANCE MATRIX...")
dist_df = pd.read_csv("country_distance_matrix_distwces.csv", index_col=0)
print(f"Distance matrix shape: {dist_df.shape}")

# ============================================================================
# PREPARE COUNTRY FEATURES
# ============================================================================

print("\nPREPARING FEATURES...")

country_features_list = []
for country in countries:
    country_data = train_df[train_df['Country Code'] == country][country_features_cols].iloc[0].values
    country_features_list.append(country_data)

country_features = np.array(country_features_list, dtype=np.float32)
country_scaler = StandardScaler()
country_features_scaled = country_scaler.fit_transform(country_features)
country_features_tensor = torch.FloatTensor(country_features_scaled).to(device)

# ============================================================================
# PREPARE INDUSTRY VARIABLES
# ============================================================================

industry_vars_list = []
for industry in industries:
    industry_data = train_df[train_df['industry_code'] == industry][industry_variables_cols].iloc[0].values
    industry_vars_list.append(industry_data)

industry_variables = np.array(industry_vars_list, dtype=np.float32)
industry_scaler = StandardScaler()
industry_variables_scaled = industry_scaler.fit_transform(industry_variables)

# ============================================================================
# PREPARE DATA LOADERS
# ============================================================================

train_country_idx = torch.LongTensor([country_to_idx[c] for c in train_df['Country Code']])
train_industry_idx = torch.LongTensor([industry_to_idx[i] for i in train_df['industry_code']])
train_target = torch.FloatTensor(train_df[target_col].values)

val_country_idx = torch.LongTensor([country_to_idx[c] for c in val_df['Country Code']])
val_industry_idx = torch.LongTensor([industry_to_idx[i] for i in val_df['industry_code']])
val_target = torch.FloatTensor(val_df[target_col].values)

test_country_idx = torch.LongTensor([country_to_idx[c] for c in test_df['Country Code']])
test_industry_idx = torch.LongTensor([industry_to_idx[i] for i in test_df['industry_code']])
test_target = torch.FloatTensor(test_df[target_col].values)

batch_size = 256
train_dataset = TensorDataset(train_country_idx, train_industry_idx, train_target)
val_dataset = TensorDataset(val_country_idx, val_industry_idx, val_target)
test_dataset = TensorDataset(test_country_idx, test_industry_idx, test_target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# FUNCTION TO CREATE ADJACENCY MATRIX
# ============================================================================

def create_adjacency_matrix(scale):
    """Create normalized adjacency matrix for given scale"""
    adj_matrix = np.zeros((num_countries, num_countries))
    
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries):
            if country1 in dist_df.index and country2 in dist_df.columns:
                dist = dist_df.loc[country1, country2]
                similarity = np.exp(-dist / scale)
                adj_matrix[i, j] = similarity
            elif i == j:
                adj_matrix[i, j] = 1.0
            else:
                adj_matrix[i, j] = 0.0
    
    # Normalize adjacency
    d = np.sum(adj_matrix, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    
    # Clip and clean
    adj_normalized = np.clip(adj_normalized, 0, 1)
    adj_normalized[adj_normalized < 1e-10] = 0
    
    return torch.FloatTensor(adj_normalized).to(device)

# ============================================================================
# FUNCTION TO TRAIN MODEL
# ============================================================================

def train_for_scale(scale, num_epochs=60, patience=15):
    """Train GCN model for a specific distance scale"""
    
    print(f"\n{'='*60}")
    print(f"Training with scale = {scale}")
    print(f"{'='*60}")
    
    # Create adjacency matrix
    adj_tensor = create_adjacency_matrix(scale)
    
    # Calculate edge density
    adj_dense = adj_tensor.cpu().numpy()
    edge_density = np.sum(adj_dense > 0) / (num_countries * num_countries)
    print(f"Edge density: {edge_density:.4f}")
    
    # Initialize model
    model = CountryIndustryPredictor(
        num_countries=num_countries,
        num_industries=num_industries,
        industry_variables=industry_variables_scaled,
        country_input_dim=len(country_features_cols),
        country_hidden_dim=64,
        country_output_dim=32,
        industry_embedding_dim=32,
        final_hidden_dim=64,
        dropout=0.2
    ).to(device)
    
    # Set graph data
    model.set_graph_data(country_features_tensor, adj_tensor)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        for batch_country, batch_industry, batch_target in train_loader:
            batch_country = batch_country.to(device)
            batch_industry = batch_industry.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_country, batch_industry)
            loss = criterion(predictions, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_losses)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for batch_country, batch_industry, batch_target in val_loader:
                batch_country = batch_country.to(device)
                batch_industry = batch_industry.to(device)
                batch_target = batch_target.to(device)
                
                predictions = model(batch_country, batch_industry)
                loss = criterion(predictions, batch_target)
                epoch_val_losses.append(loss.item())
        
        avg_val_loss = np.mean(epoch_val_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
    
    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.2f} seconds")
    print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    # Evaluate on validation set
    val_preds = []
    val_targets_list = []
    with torch.no_grad():
        for batch_country, batch_industry, batch_target in val_loader:
            batch_country = batch_country.to(device)
            batch_industry = batch_industry.to(device)
            pred = model(batch_country, batch_industry)
            val_preds.extend(pred.cpu().numpy())
            val_targets_list.extend(batch_target.numpy())
    
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets_list)
    
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_r2 = r2_score(val_targets, val_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)
    
    # Evaluate on test set
    test_preds = []
    test_targets_list = []
    with torch.no_grad():
        for batch_country, batch_industry, batch_target in test_loader:
            batch_country = batch_country.to(device)
            batch_industry = batch_industry.to(device)
            pred = model(batch_country, batch_industry)
            test_preds.extend(pred.cpu().numpy())
            test_targets_list.extend(batch_target.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets_list)
    
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    
    print(f"\n  Validation - RMSE: {val_rmse:.6f}, R²: {val_r2:.6f}")
    print(f"  Test - RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")
    
    return {
        'scale': scale,
        'edge_density': edge_density,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_mae': val_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'best_val_loss': best_val_loss
    }

# ============================================================================
# RUN TUNING
# ============================================================================

print("\n" + "-" * 40)
print("STARTING HYPERPARAMETER TUNING")
print("-" * 40)

scales = [5000, 1000, 500, 200, 100, 50]
print(f"Testing scales: {scales}")

# Store results
tuning_results = []

for scale in scales:
    try:
        result = train_for_scale(scale, num_epochs=100, patience=12)
        tuning_results.append(result)
    except Exception as e:
        print(f"Error with scale {scale}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# COMPILE AND SAVE RESULTS
# ============================================================================

print("\n" + "-" * 40)
print("COMPILING RESULTS")
print("-" * 40)

# Create summary DataFrame
summary_df = pd.DataFrame(tuning_results)
summary_df = summary_df.sort_values('scale', ascending=False)

print("\n" + "=" * 80)
print("TUNING SUMMARY")
print("=" * 80)
print(summary_df[['scale', 'val_rmse', 'val_r2', 'test_rmse', 'test_r2', 'training_time']].to_string())
print("=" * 80)

# Find best scale
best_idx = summary_df['val_r2'].idxmax()
best_scale = summary_df.loc[best_idx, 'scale']
best_val_r2 = summary_df.loc[best_idx, 'val_r2']
best_test_r2 = summary_df.loc[best_idx, 'test_r2']
best_val_rmse = summary_df.loc[best_idx, 'val_rmse']
best_test_rmse = summary_df.loc[best_idx, 'test_rmse']

print(f"\n{'='*60}")
print(f"BEST SCALE: {best_scale}")
print(f"  Validation RMSE: {best_val_rmse:.6f}")
print(f"  Validation R²: {best_val_r2:.6f}")
print(f"  Test RMSE: {best_test_rmse:.6f}")
print(f"  Test R²: {best_test_r2:.6f}")
print(f"{'='*60}")

# Save results
summary_df.to_csv('../results/tuning/scale_tuning_results.csv', index=False)

# Save JSON
results_json = {
    'scales_tested': scales,
    'best_scale': best_scale,
    'best_val_r2': best_val_r2,
    'best_test_r2': best_test_r2,
    'best_val_rmse': best_val_rmse,
    'best_test_rmse': best_test_rmse,
    'results': tuning_results
}

with open('../results/tuning/scale_tuning_results.json', 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\nResults saved to: ../results/tuning/")

# ============================================================================
# PLOT RESULTS
# ============================================================================

print("\n" + "-" * 40)
print("CREATING PLOT")
print("-" * 40)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

scales_plot = summary_df['scale'].values
val_r2 = summary_df['val_r2'].values
test_r2 = summary_df['test_r2'].values
val_rmse = summary_df['val_rmse'].values
test_rmse = summary_df['test_rmse'].values

# Plot R²
ax1.plot(scales_plot, val_r2, 'o-', label='Validation R²', linewidth=2, markersize=8, color='blue')
ax1.plot(scales_plot, test_r2, 's-', label='Test R²', linewidth=2, markersize=8, color='green')
ax1.set_xscale('log')
ax1.set_xlabel('Distance Scale')
ax1.set_ylabel('R² Score')
ax1.set_title('R² vs Distance Scale')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best_scale, color='r', linestyle='--', alpha=0.5, linewidth=2)

# Plot RMSE
ax2.plot(scales_plot, val_rmse, 'o-', label='Validation RMSE', linewidth=2, markersize=8, color='blue')
ax2.plot(scales_plot, test_rmse, 's-', label='Test RMSE', linewidth=2, markersize=8, color='green')
ax2.set_xscale('log')
ax2.set_xlabel('Distance Scale')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE vs Distance Scale')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axvline(x=best_scale, color='r', linestyle='--', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('../results/tuning/scale_tuning_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to: ../results/tuning/scale_tuning_plot.png")

# ============================================================================
# FINAL REPORT
# ============================================================================

report = f"""
================================================================================
GCN DISTANCE SCALE TUNING REPORT
================================================================================

Scales Tested: {scales}

Best Scale: {best_scale}
  - Validation RMSE: {best_val_rmse:.6f}
  - Validation R²: {best_val_r2:.6f}
  - Test RMSE: {best_test_rmse:.6f}
  - Test R²: {best_test_r2:.6f}

Detailed Results:
{summary_df[['scale', 'val_rmse', 'val_r2', 'test_rmse', 'test_r2', 'training_time']].to_string()}

================================================================================
RECOMMENDATION
================================================================================
Based on validation R², the optimal distance scale is {best_scale}.

You should update line 169 in the original script (07_gcn.py) to:
scale = {best_scale}  # Your optimized scale

================================================================================
"""

print(report)

with open('../results/tuning/tuning_report.txt', 'w') as f:
    f.write(report)

print("\n" + "=" * 60)
print("TUNING COMPLETE!")
print("=" * 60)