"""
Script 11: GCN with Industry Variables for Carbon Intensity Prediction
Uses: train.csv, val.csv, test.csv, country_distance_matrix_distwces.csv, industry variables
Saves: gcn_industry_vars_results.json, predictions, visualizations
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
print("GCN WITH INDUSTRY VARIABLES SCRIPT")
print("=" * 60)
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# GCN LAYER IMPLEMENTATION (same as before)
# ============================================================================

class GraphConvLayer(nn.Module):
    """Simple Graph Convolution layer"""
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # Graph convolution: H' = σ(A @ H @ W)
        support = torch.mm(adj, x)  # [num_nodes, in_features]
        out = self.linear(support)  # [num_nodes, out_features]
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
        # First GCN layer
        x = self.gcn1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.gcn2(x, adj)
        
        return x

# ============================================================================
# NEW: Industry Encoder - Uses your 8 variables to create 32-dim embeddings
# ============================================================================

class IndustryEncoder(nn.Module):
    """
    Creates industry embeddings from the 8 industry variables.
    This preserves your expert knowledge while allowing learning.
    """
    def __init__(self, num_industries, industry_variables, embedding_dim=32):
        super(IndustryEncoder, self).__init__()
        
        # Store industry variables as a buffer (not trainable, but on correct device)
        self.register_buffer('industry_vars', 
                           torch.FloatTensor(industry_variables))  # [400 × 8]
        
        # Learnable projection: 8 dimensions → 32 dimensions
        # This transforms your expert variables into a richer representation
        self.projection = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim)
        )
        
        # Optional: Small learnable residual for fine-tuning
        # This allows the model to adjust the embeddings based on data
        self.residual = nn.Embedding(num_industries, embedding_dim)
        
        # Initialize residual to near zero (start with pure projection)
        nn.init.zeros_(self.residual.weight)
        
    def forward(self, industry_idx):
        """
        Args:
            industry_idx: [batch_size] indices of industries
        Returns:
            industry_embeddings: [batch_size × embedding_dim]
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

class CountryIndustryPredictor(nn.Module):
    """Complete model: GCN for countries + IndustryEncoder for industries"""
    def __init__(self, num_countries, num_industries, industry_variables,
                 country_input_dim, country_hidden_dim=64, country_output_dim=32,
                 industry_embedding_dim=32, final_hidden_dim=64, dropout=0.2):
        super(CountryIndustryPredictor, self).__init__()
        
        # GCN for countries (same as before)
        self.gcn = GCN(
            input_dim=country_input_dim,
            hidden_dim=country_hidden_dim,
            output_dim=country_output_dim,
            dropout=dropout
        )
        
        # NEW: Industry encoder using your 8 variables
        self.industry_encoder = IndustryEncoder(
            num_industries=num_industries,
            industry_variables=industry_variables,
            embedding_dim=industry_embedding_dim
        )
        
        # Store country features as parameter (will be set later)
        self.country_features = None
        self.adjacency = None
        self.country_embeddings = None
        
        # Final prediction network (takes concatenated country + industry embeddings)
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
        """Store graph data and pre-compute country embeddings"""
        self.country_features = country_features
        self.adjacency = adjacency
        # Pre-compute country embeddings
        with torch.no_grad():
            self.country_embeddings = self.gcn(country_features, adjacency)
        
    def forward(self, country_idx, industry_idx):
        """
        Args:
            country_idx: [batch_size] indices of countries
            industry_idx: [batch_size] indices of industries
        """
        # Get country embeddings (pre-computed)
        country_emb = self.country_embeddings[country_idx]  # [batch × 32]
        
        # NEW: Get industry embeddings from your 8 variables
        industry_emb = self.industry_encoder(industry_idx)  # [batch × 32]
        
        # Concatenate
        combined = torch.cat([country_emb, industry_emb], dim=1)  # [batch × 64]
        
        # Predict
        output = self.predictor(combined)
        
        return output.squeeze()

# ============================================================================
# DEFINE METRICS FUNCTIONS (same as before)
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
# LOAD DATA (same as before)
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

# IMPORTANT: Remove geographic features from country input (let GCN handle geography)
country_features_cols = [col for col in feature_cols if col in 
                         ['gdp_per_capita_ppp', 'industry_value_added_pct', 
                          'renewable_energy_pct', 'coal_electricity_pct',
                          'energy_intensity_level', 'gdp_per_energy_unit',
                          'urban_population_pct', 'natural_resources_rents_pct']]

# Industry variables (your 8 scores)
industry_variables_cols = [col for col in feature_cols if col in
                          ['process_emission_intensity_score', 'material_processing_depth_score',
                           'thermal_process_intensity_score', 'electrification_feasibility_score',
                           'continuous_operations_intensity_score', 'material_throughput_scale_score',
                           'chemical_intensity_score', 'capital_vs_labor_intensity_score']]

print(f"\nCountry features ({len(country_features_cols)}):")
for col in country_features_cols:
    print(f"  {col}")

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
# LOAD DISTANCE MATRIX AND CREATE ADJACENCY (same as before, with scale=100)
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DISTANCE MATRIX")
print("-" * 40)

# Load distance matrix
dist_df = pd.read_csv("country_distance_matrix_distwces.csv", index_col=0)

# Create adjacency matrix
adj_matrix = np.zeros((num_countries, num_countries))
scale = 100  # Your optimized scale

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

# Normalize adjacency (symmetric normalization)
d = np.sum(adj_matrix, axis=1)
d_inv_sqrt = np.power(d, -0.5)
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
d_inv_sqrt = np.diag(d_inv_sqrt)
adj_normalized = d_inv_sqrt @ adj_matrix @ d_inv_sqrt

# FIX: Clip negative values from numerical precision
adj_normalized = np.clip(adj_normalized, 0, 1)
adj_normalized[adj_normalized < 1e-10] = 0

# Convert to tensor
adj_tensor = torch.FloatTensor(adj_normalized).to(device)

print(f"\nAdjacency matrix shape: {adj_tensor.shape}")
print(f"Edge density: {np.sum(adj_matrix > 0) / (num_countries * num_countries):.4f}")
print(f"Min value after clipping: {adj_normalized.min():.2e}")
print(f"Negative entries after clipping: {(adj_normalized < 0).sum()}")

# ============================================================================
# PREPARE COUNTRY FEATURES (only WDI indicators, no geography)
# ============================================================================

print("\n" + "-" * 40)
print("PREPARING COUNTRY FEATURES")
print("-" * 40)

# Aggregate country features from training data
country_features_list = []
for country in countries:
    country_data = train_df[train_df['Country Code'] == country][country_features_cols].iloc[0].values
    country_features_list.append(country_data)

country_features = np.array(country_features_list, dtype=np.float32)

# Scale country features
country_scaler = StandardScaler()
country_features_scaled = country_scaler.fit_transform(country_features)

# Convert to tensor
country_features_tensor = torch.FloatTensor(country_features_scaled).to(device)

print(f"Country features shape: {country_features_tensor.shape}")

# ============================================================================
# NEW: PREPARE INDUSTRY VARIABLES (your 8 scores)
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

# Scale industry variables (optional, but helps with projection)
industry_scaler = StandardScaler()
industry_variables_scaled = industry_scaler.fit_transform(industry_variables)

print(f"Industry variables shape: {industry_variables_scaled.shape}")
print(f"Industry variables stats:")
print(f"  Mean: {industry_variables_scaled.mean(axis=0).round(3)}")
print(f"  Std:  {industry_variables_scaled.std(axis=0).round(3)}")

# Save scaler for later use
joblib.dump(industry_scaler, '../models/gcn_industry_scaler.pkl')
print("Industry scaler saved to: ../models/gcn_industry_scaler.pkl")

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

# Create data loaders
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
# INITIALIZE MODEL WITH INDUSTRY VARIABLES
# ============================================================================

print("\n" + "-" * 40)
print("INITIALIZING MODEL WITH INDUSTRY VARIABLES")
print("-" * 40)

model = CountryIndustryPredictor(
    num_countries=num_countries,
    num_industries=num_industries,
    industry_variables=industry_variables_scaled,  # Pass your 8 variables!
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

# Count parameters in industry encoder
industry_params = sum(p.numel() for p in model.industry_encoder.parameters())
print(f"Industry encoder parameters: {industry_params:,} (from your 8 variables)")

# ============================================================================
# TRAINING SETUP (same as before)
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
# TRAINING LOOP (same as before)
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
        batch_country = batch_country.to(device)
        batch_industry = batch_industry.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_country, batch_industry)
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
            batch_country = batch_country.to(device)
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
            'country_features_cols': country_features_cols,
            'industry_variables_cols': industry_variables_cols,
            'countries': countries,
            'industries': industries,
            'country_to_idx': country_to_idx,
            'industry_to_idx': industry_to_idx
        }, '../models/gcn_industry_vars_best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

# Save scalers
joblib.dump(country_scaler, '../models/gcn_industry_vars_country_scaler.pkl')
print("Country scaler saved to: ../models/gcn_industry_vars_country_scaler.pkl")

# ============================================================================
# GENERATE PREDICTIONS (same as before)
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
        batch_country = batch_country.to(device)
        batch_industry = batch_industry.to(device)
        pred = model(batch_country, batch_industry)
        predictions['train'].extend(pred.cpu().numpy())
        targets['train'].extend(batch_target.numpy())
    
    # Validation predictions
    for batch_country, batch_industry, batch_target in val_loader:
        batch_country = batch_country.to(device)
        batch_industry = batch_industry.to(device)
        pred = model(batch_country, batch_industry)
        predictions['val'].extend(pred.cpu().numpy())
        targets['val'].extend(batch_target.numpy())
    
    # Test predictions
    for batch_country, batch_industry, batch_target in test_loader:
        batch_country = batch_country.to(device)
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
# EVALUATE ON ALL SETS (same as before)
# ============================================================================

print("\n" + "-" * 40)
print("EVALUATION RESULTS")
print("-" * 40)

train_metrics = evaluate_predictions(y_train, y_pred_train, "TRAIN SET")
val_metrics = evaluate_predictions(y_val, y_pred_val, "VALIDATION SET")
test_metrics = evaluate_predictions(y_test, y_pred_test, "TEST SET")

# ============================================================================
# RESIDUAL ANALYSIS (same as before)
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
# SAVE PREDICTIONS (same as before)
# ============================================================================

print("\n" + "-" * 40)
print("SAVING PREDICTIONS")
print("-" * 40)

# Save test predictions
test_results_df.to_csv('../results/gcn_industry_vars_test_predictions.csv', index=False)
print("Test predictions saved to: ../results/gcn_industry_vars_test_predictions.csv")

# Save train predictions
train_results_df = train_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
train_results_df['predicted'] = y_pred_train
train_results_df['residual'] = y_train - y_pred_train
train_results_df['abs_error'] = np.abs(train_results_df['residual'])
train_results_df.to_csv('../results/gcn_industry_vars_train_predictions.csv', index=False)
print("Train predictions saved to: ../results/gcn_industry_vars_train_predictions.csv")

# Save validation predictions
val_results_df = val_df[['Country', 'Country Code', 'industry_code', 'industry_name', 'carbon_intensity']].copy()
val_results_df['predicted'] = y_pred_val
val_results_df['residual'] = y_val - y_pred_val
val_results_df['abs_error'] = np.abs(val_results_df['residual'])
val_results_df.to_csv('../results/gcn_industry_vars_val_predictions.csv', index=False)
print("Validation predictions saved to: ../results/gcn_industry_vars_val_predictions.csv")

# Summary statistics
summary_stats = pd.DataFrame({
    'dataset': ['Train', 'Validation', 'Test'],
    'count': [len(y_train), len(y_val), len(y_test)],
    'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
    'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
    'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
    'MAPE': [train_metrics['MAPE'], val_metrics['MAPE'], test_metrics['MAPE']]
})
summary_stats.to_csv('../results/gcn_industry_vars_summary_stats.csv', index=False)
print("Summary statistics saved to: ../results/gcn_industry_vars_summary_stats.csv")

# ============================================================================
# VISUALIZATIONS (same as before, with updated titles)
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
axes[0, 0].set_title(f'GCN+IndustryVars: Predicted vs Actual (R² = {test_metrics["R2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals Distribution
axes[0, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].axvline(x=np.mean(residuals_test), color='g', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(residuals_test):.4f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('GCN+IndustryVars: Residual Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Training History
axes[0, 2].plot(history['epoch'], history['train_loss'], label='Train', alpha=0.7)
axes[0, 2].plot(history['epoch'], history['val_loss'], label='Validation', alpha=0.7)
axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss (MSE)')
axes[0, 2].set_title('GCN+IndustryVars: Training History')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Absolute Error Distribution
axes[1, 0].hist(np.abs(residuals_test), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=test_metrics['MAE'], color='r', linestyle='--', lw=2, 
                   label=f'MAE: {test_metrics["MAE"]:.4f}')
axes[1, 0].set_xlabel('Absolute Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('GCN+IndustryVars: Absolute Error Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Metrics Comparison
metrics_compare = pd.DataFrame({
    'Train': [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['R2']],
    'Validation': [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['R2']],
    'Test': [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['R2']]
}, index=['MAE', 'RMSE', 'R²'])

metrics_compare.T.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('GCN+IndustryVars: Metrics Across Datasets')
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
axes[1, 2].set_title('GCN+IndustryVars: Log Scale Comparison')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/gcn_industry_vars_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# ADD THIS SECTION - COUNTRY EMBEDDING SIMILARITY ANALYSIS
# ============================================================================

print("\n" + "-" * 40)
print("ANALYZING COUNTRY EMBEDDINGS")
print("-" * 40)

# Get country embeddings
with torch.no_grad():
    country_embeddings = model.gcn(country_features_tensor, adj_tensor).cpu().numpy()

# Compute similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
country_similarity = cosine_similarity(country_embeddings)

# Select representative countries to display
example_countries = ['USA', 'CHN', 'DEU', 'BRA', 'IND', 'ZAF', 'SAU', 'AUS', 'GBR', 'JPN']
print("\nMost similar countries based on GCN embeddings:")

for country in example_countries:
    if country in country_to_idx:
        idx = country_to_idx[country]
        similarities = country_similarity[idx]
        
        # Get top 5 most similar (excluding self)
        most_similar = np.argsort(similarities)[-6:-1][::-1]
        
        print(f"\n{country} most similar to:")
        for sim_idx in most_similar:
            sim_country = countries[sim_idx]
            sim_score = similarities[sim_idx]
            print(f"  {sim_country}: {sim_score:.4f}")

# Create similarity matrix for a subset (for visualization)
subset_countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'IND', 'BRA', 'CAN', 'AUS', 
                    'RUS', 'ZAF', 'SAU', 'MEX', 'IDN']
subset_indices = [country_to_idx[c] for c in subset_countries if c in country_to_idx]
subset_names = [countries[i] for i in subset_indices]
subset_similarity = country_similarity[np.ix_(subset_indices, subset_indices)]

# Plot similarity heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(subset_similarity, xticklabels=subset_names, yticklabels=subset_names,
            annot=True, fmt='.3f', cmap='viridis', ax=ax, 
            vmin=0, vmax=1)  # Fix range [0,1] for similarities
ax.set_title('Country Embedding Similarities (GCN + Industry Variables)')
plt.tight_layout()
plt.savefig('../results/plots/gcn_industry_vars_country_similarities.png', dpi=150, bbox_inches='tight')
plt.show()

# Also save full similarity matrix for later analysis
similarity_df = pd.DataFrame(country_similarity, 
                             index=countries, 
                             columns=countries)
similarity_df.to_csv('../results/gcn_industry_vars_country_similarities.csv')
print("\nFull country similarity matrix saved to: ../results/gcn_industry_vars_country_similarities.csv")

# Find most and least similar pairs
print("\nMost similar country pairs (excluding self):")
similarities_flat = []
for i in range(len(countries)):
    for j in range(i+1, len(countries)):
        similarities_flat.append((countries[i], countries[j], country_similarity[i,j]))

similarities_flat.sort(key=lambda x: x[2], reverse=True)
for i in range(10):
    c1, c2, sim = similarities_flat[i]
    print(f"  {c1} - {c2}: {sim:.4f}")

print("\nLeast similar country pairs:")
for i in range(10):
    c1, c2, sim = similarities_flat[-(i+1)]
    print(f"  {c1} - {c2}: {sim:.4f}")

# ============================================================================
# OPTIONAL: Visualize country embeddings in 2D using PCA
# ============================================================================

from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
country_2d = pca.fit_transform(country_embeddings)

# Create DataFrame for plotting
plot_df = pd.DataFrame({
    'x': country_2d[:, 0],
    'y': country_2d[:, 1],
    'country': countries
})

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
scatter = ax.scatter(plot_df['x'], plot_df['y'], alpha=0.6, s=50)

# Label selected countries
for country in example_countries:
    if country in country_to_idx:
        row = plot_df[plot_df['country'] == country].iloc[0]
        ax.annotate(country, (row['x'], row['y']), fontsize=9, alpha=0.8)

ax.set_title('Country Embeddings Visualized in 2D (PCA)')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/plots/gcn_industry_vars_country_pca.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ============================================================================
# ADDITIONAL ANALYSIS: Industry Embedding Visualization
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
            emb = model.industry_encoder(torch.LongTensor([idx]).to(device)).cpu().numpy()[0]
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
    ax.set_title('Industry Embedding Similarities (from your 8 variables)')
    plt.tight_layout()
    plt.savefig('../results/plots/gcn_industry_similarities.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nIndustry similarities based on your variables + learning:")
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
        'country_hidden_dim': 64,
        'country_output_dim': 32,
        'industry_embedding_dim': 32,
        'final_hidden_dim': 64,
        'dropout': 0.2,
        'total_params': total_params,
        'industry_encoder_params': industry_params
    },
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'training_history': history,
    'best_epoch': best_epoch,
    'training_time': training_time
}

with open('../results/gcn_industry_vars_results.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("All metrics saved to: ../results/gcn_industry_vars_results.json")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 60)
print("GCN WITH INDUSTRY VARIABLES - FINAL SUMMARY")
print("=" * 60)
print(f"Model Parameters: {total_params:,}")
print(f"  - Industry Encoder: {industry_params:,} (from your 8 variables)")
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
print(f"MedAE: {test_metrics['MedAE']:.6f}")
print("=" * 60)
print(f"Results saved to: ../results/")
print(f"Plots saved to: ../results/plots/")
print("=" * 60)