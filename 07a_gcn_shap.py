"""
Script 12: GCN with Industry Variables + SHAP Explainability
Uses: trained model from script 11, test data
Saves: SHAP explanations, visualizations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path("../results/shap").mkdir(parents=True, exist_ok=True)
Path("../results/plots/shap").mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("GCN WITH SHAP EXPLAINABILITY")
print("=" * 60)
print(f"Using device: {device}")

# ============================================================================
# LOAD MODEL AND DATA (same architecture as script 11)
# ============================================================================

# Define model classes (must match training)
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        support = torch.mm(adj, x)
        out = self.linear(support)
        return out

class GCN(nn.Module):
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
    def __init__(self, num_industries, industry_variables, embedding_dim=32):
        super(IndustryEncoder, self).__init__()
        self.register_buffer('industry_vars', torch.FloatTensor(industry_variables))
        
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
        return projected + 0.1 * residual

class CountryIndustryPredictor(nn.Module):
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
        
        self.predictor = nn.Sequential(
            nn.Linear(country_output_dim + industry_embedding_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim // 2, 1)
        )
        
        self.country_embeddings = None
        
    def set_graph_data(self, country_features, adjacency):
        with torch.no_grad():
            self.country_embeddings = self.gcn(country_features, adjacency)
        
    def forward(self, country_idx, industry_idx):
        country_emb = self.country_embeddings[country_idx]
        industry_emb = self.industry_encoder(industry_idx)
        combined = torch.cat([country_emb, industry_emb], dim=1)
        return self.predictor(combined).squeeze()

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

print("\n" + "-" * 40)
print("LOADING DATA AND MODEL")
print("-" * 40)

# Load test data
test_df = pd.read_csv("../data/test.csv")
print(f"Test set: {len(test_df):,} rows")

# Load feature info
with open('../data/feature_info.json', 'r') as f:
    feature_info = json.load(f)

feature_cols = feature_info['feature_names']

# Define column groups
country_features_cols = ['gdp_per_capita_ppp', 'industry_value_added_pct', 
                         'renewable_energy_pct', 'coal_electricity_pct',
                         'energy_intensity_level', 'gdp_per_energy_unit',
                         'urban_population_pct', 'natural_resources_rents_pct']

industry_variables_cols = ['process_emission_intensity_score', 'material_processing_depth_score',
                           'thermal_process_intensity_score', 'electrification_feasibility_score',
                           'continuous_operations_intensity_score', 'material_throughput_scale_score',
                           'chemical_intensity_score', 'capital_vs_labor_intensity_score']

# Get unique countries and industries
countries = sorted(test_df['Country Code'].unique())
industries = sorted(test_df['industry_code'].unique())

country_to_idx = {country: i for i, country in enumerate(countries)}
industry_to_idx = {industry: i for i, industry in enumerate(industries)}

num_countries = len(countries)
num_industries = len(industries)

print(f"Number of countries: {num_countries}")
print(f"Number of industries: {num_industries}")

# Load scalers and prepare data
country_scaler = joblib.load('../models/gcn_industry_vars_country_scaler.pkl')
industry_scaler = joblib.load('../models/gcn_industry_scaler.pkl')

# Prepare country features
country_features_list = []
for country in countries:
    country_data = test_df[test_df['Country Code'] == country][country_features_cols].iloc[0].values
    country_features_list.append(country_data)
country_features = np.array(country_features_list, dtype=np.float32)
country_features_scaled = country_scaler.transform(country_features)
country_features_tensor = torch.FloatTensor(country_features_scaled).to(device)

# Prepare industry variables
industry_vars_list = []
for industry in industries:
    industry_data = test_df[test_df['industry_code'] == industry][industry_variables_cols].iloc[0].values
    industry_vars_list.append(industry_data)
industry_variables = np.array(industry_vars_list, dtype=np.float32)
industry_variables_scaled = industry_scaler.transform(industry_variables)

# Load adjacency matrix
dist_df = pd.read_csv("country_distance_matrix_distwces.csv", index_col=0)
adj_matrix = np.zeros((num_countries, num_countries))
scale = 1000

for i, country1 in enumerate(countries):
    for j, country2 in enumerate(countries):
        if country1 in dist_df.index and country2 in dist_df.columns:
            dist = dist_df.loc[country1, country2]
            similarity = np.exp(-dist / scale)
            adj_matrix[i, j] = similarity
        elif i == j:
            adj_matrix[i, j] = 1.0

# Normalize adjacency
d = np.sum(adj_matrix, axis=1)
d_inv_sqrt = np.power(d, -0.5)
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
d_inv_sqrt = np.diag(d_inv_sqrt)
adj_normalized = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
adj_normalized = np.clip(adj_normalized, 0, 1)
adj_tensor = torch.FloatTensor(adj_normalized).to(device)

# Load model
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

checkpoint = torch.load('../models/gcn_industry_vars_best_model.pt', 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.set_graph_data(country_features_tensor, adj_tensor)
model.eval()

print("Model loaded successfully!")

# ============================================================================
# HELPER FUNCTION: Get prediction for a specific country-industry pair
# ============================================================================

def predict_with_features(country_idx, industry_idx, 
                          country_feature_perturb=None, 
                          industry_var_perturb=None):
    """
    Make prediction with optional feature perturbations
    """
    with torch.no_grad():
        # Get country embedding (using original or perturbed features)
        if country_feature_perturb is not None:
            # Perturb country features for this specific country
            X_pert = country_features_tensor.clone()
            X_pert[country_idx] = torch.FloatTensor(country_feature_perturb).to(device)
            # Recompute GCN for this perturbation
            h1 = torch.relu(model.gcn.gcn1(X_pert, adj_tensor))
            h1 = model.gcn.dropout(h1)
            h2 = model.gcn.gcn2(h1, adj_tensor)
            country_emb = h2[country_idx:country_idx+1]
        else:
            country_emb = model.country_embeddings[country_idx:country_idx+1]
        
        # Get industry embedding (using original or perturbed variables)
        if industry_var_perturb is not None:
            # Create temporary tensor with perturbed variables
            temp_vars = model.industry_encoder.industry_vars.clone()
            temp_vars[industry_idx] = torch.FloatTensor(industry_var_perturb).to(device)
            
            # Compute industry embedding
            projected = model.industry_encoder.projection(temp_vars[industry_idx:industry_idx+1])
            residual = model.industry_encoder.residual(torch.LongTensor([industry_idx]).to(device))
            industry_emb = projected + 0.1 * residual
        else:
            industry_emb = model.industry_encoder(torch.LongTensor([industry_idx]).to(device))
            if len(industry_emb.shape) == 1:
                industry_emb = industry_emb.unsqueeze(0)
        
        # Combine and predict
        combined = torch.cat([country_emb, industry_emb], dim=1)
        prediction = model.predictor(combined).squeeze()
        
    return prediction.cpu().numpy()

# ============================================================================
# PART 1: INDUSTRY VARIABLES SHAP (Your 8 variables)
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: EXPLAINING INDUSTRY VARIABLES")
print("=" * 60)

class IndustryVarsWrapper:
    """Wrapper to explain how industry variables affect predictions"""
    def __init__(self, model, country_idx, industry_idx, 
                 country_features_scaled, industry_vars_scaled):
        self.model = model
        self.country_idx = country_idx
        self.industry_idx = industry_idx
        self.country_features_scaled = country_features_scaled
        self.industry_vars_scaled = industry_vars_scaled
        
    def __call__(self, X):
        """
        X: perturbations of the 8 industry variables [n_samples, 8]
        """
        predictions = []
        for pert in X:
            pred = predict_with_features(
                self.country_idx, self.industry_idx,
                country_feature_perturb=None,
                industry_var_perturb=pert
            )
            predictions.append(pred)
        return np.array(predictions)

# Select example pairs to explain
example_pairs = [
    ('EGY', '327310', 'Egypt + Cement'),           # High outlier
    ('USA', '327310', 'USA + Cement'),             # Normal
    ('DEU', '336111', 'Germany + Auto'),           # Industrial
    ('CHN', '334111', 'China + Electronics'),      # Tech
    ('BRA', '311111', 'Brazil + Pet Food')         # Agriculture-related
]

print("\n" + "-" * 40)
print("SHAP Analysis for Industry Variables")
print("-" * 40)

for country_code, ind_code, pair_name in example_pairs:
    if country_code in country_to_idx and ind_code in industry_to_idx:
        print(f"\n📊 Analyzing {pair_name}...")
        
        c_idx = country_to_idx[country_code]
        i_idx = industry_to_idx[ind_code]
        
        # Get the industry variables for this industry
        industry_vars = industry_variables_scaled[i_idx:i_idx+1]
        
        # Create wrapper
        wrapper = IndustryVarsWrapper(
            model, c_idx, i_idx,
            country_features_scaled, industry_variables_scaled
        )
        
        # Use smaller background for speed
        background = industry_variables_scaled[np.random.choice(len(industries), 50, replace=False)]
        
        # Create explainer
        explainer = shap.KernelExplainer(wrapper, background)
        
        # Calculate SHAP values (explaining the prediction, not the embedding)
        shap_values = explainer.shap_values(industry_vars, nsamples=50)
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, industry_vars, 
                         feature_names=industry_variables_cols,
                         show=False)
        plt.title(f'SHAP Values for {pair_name}\n(Impact of industry variables on prediction)')
        plt.tight_layout()
        plt.savefig(f'../results/plots/shap/industry_{country_code}_{ind_code}_shap.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = sorted(zip(industry_variables_cols, mean_abs_shap), 
                            key=lambda x: x[1], reverse=True)
        print(f"  Top influential variables for {pair_name}:")
        for feat, val in top_features[:3]:
            print(f"    • {feat}: {val:.4f}")

# ============================================================================
# PART 2: COUNTRY FEATURES SHAP
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: EXPLAINING COUNTRY FEATURES")
print("=" * 60)

class CountryFeaturesWrapper:
    """Wrapper to explain how country features affect predictions"""
    def __init__(self, model, country_idx, industry_idx,
                 country_features_scaled, industry_vars_scaled):
        self.model = model
        self.country_idx = country_idx
        self.industry_idx = industry_idx
        self.country_features_scaled = country_features_scaled
        self.industry_vars_scaled = industry_vars_scaled
        
    def __call__(self, X):
        """
        X: perturbations of the 8 country features [n_samples, 8]
        """
        predictions = []
        for pert in X:
            pred = predict_with_features(
                self.country_idx, self.industry_idx,
                country_feature_perturb=pert,
                industry_var_perturb=None
            )
            predictions.append(pred)
        return np.array(predictions)

print("\n" + "-" * 40)
print("SHAP Analysis for Country Features")
print("-" * 40)

for country_code, ind_code, pair_name in example_pairs[:3]:  # First 3 for speed
    if country_code in country_to_idx and ind_code in industry_to_idx:
        print(f"\n📊 Analyzing {pair_name}...")
        
        c_idx = country_to_idx[country_code]
        i_idx = industry_to_idx[ind_code]
        
        # Get the country features
        country_feats = country_features_scaled[c_idx:c_idx+1]
        
        # Create wrapper
        wrapper = CountryFeaturesWrapper(
            model, c_idx, i_idx,
            country_features_scaled, industry_variables_scaled
        )
        
        # Background
        background = country_features_scaled[np.random.choice(len(countries), 50, replace=False)]
        
        # Create explainer
        explainer = shap.KernelExplainer(wrapper, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(country_feats, nsamples=50)
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, country_feats, 
                         feature_names=country_features_cols,
                         show=False)
        plt.title(f'SHAP Values for {pair_name}\n(Impact of country features on prediction)')
        plt.tight_layout()
        plt.savefig(f'../results/plots/shap/country_{country_code}_{ind_code}_shap.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = sorted(zip(country_features_cols, mean_abs_shap), 
                            key=lambda x: x[1], reverse=True)
        print(f"  Top influential country features for {pair_name}:")
        for feat, val in top_features[:3]:
            print(f"    • {feat}: {val:.4f}")

# ============================================================================
# PART 3: GLOBAL ANALYSIS (Most important features across all predictions)
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

print("\n" + "-" * 40)
print("Computing global SHAP values on test set sample...")
print("-" * 40)

# Take a sample of test data
test_sample = test_df.sample(min(200, len(test_df)), random_state=42)

# Prepare data for global analysis
country_idxs = [country_to_idx[c] for c in test_sample['Country Code']]
industry_idxs = [industry_to_idx[i] for i in test_sample['industry_code']]

# Create a combined feature matrix for global analysis
# We'll use the original 8 + 8 = 16 features
global_features = []
global_predictions = []

for i, (c_idx, i_idx) in enumerate(zip(country_idxs, industry_idxs)):
    # Get original features
    country_feats = country_features_scaled[c_idx]
    industry_vars = industry_variables_scaled[i_idx]
    
    # Combine
    combined_feats = np.concatenate([country_feats, industry_vars])
    global_features.append(combined_feats)
    
    # Get prediction
    with torch.no_grad():
        pred = predict_with_features(c_idx, i_idx)
    global_predictions.append(pred)

global_features = np.array(global_features)
feature_names = country_features_cols + industry_variables_cols

# Global wrapper
def global_predict(X):
    """X: [n_samples, 16] combined features"""
    predictions = []
    for features in X:
        country_feats = features[:8]
        industry_vars = features[8:]
        
        # We need to find which country/industry this corresponds to
        # This is approximate - we find closest match
        c_idx = np.argmin(np.sum((country_features_scaled - country_feats)**2, axis=1))
        i_idx = np.argmin(np.sum((industry_variables_scaled - industry_vars)**2, axis=1))
        
        pred = predict_with_features(c_idx, i_idx)
        predictions.append(pred)
    return np.array(predictions)

# Background
background_global = global_features[np.random.choice(len(global_features), 50, replace=False)]

# Create explainer
explainer = shap.KernelExplainer(global_predict, background_global)

# Calculate SHAP
shap_values_global = explainer.shap_values(global_features[:100], nsamples=50)

# Plot global importance
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_global, global_features[:100], 
                 feature_names=feature_names,
                 show=False, max_display=20)
plt.title('Global Feature Importance\n(Impact on predictions across all samples)')
plt.tight_layout()
plt.savefig('../results/plots/shap/global_feature_importance.png', 
           dpi=150, bbox_inches='tight')
plt.show()

# Print top features
mean_abs_shap_global = np.abs(shap_values_global).mean(axis=0)
top_global = sorted(zip(feature_names, mean_abs_shap_global), 
                   key=lambda x: x[1], reverse=True)
print("\nTop 10 most important features globally:")
for feat, val in top_global[:10]:
    print(f"  • {feat}: {val:.4f}")

# ============================================================================
# PART 4: WATERFALL PLOTS FOR SPECIFIC PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: DETAILED WATERFALL PLOTS")
print("=" * 60)

# Select a few interesting predictions
interesting_pairs = [
    ('EGY', '327310', 'Egypt + Cement (High outlier)'),
    ('USA', '327310', 'USA + Cement (Normal)'),
    ('DEU', '336111', 'Germany + Auto (Industrial)')
]

for country_code, ind_code, title in interesting_pairs:
    if country_code in country_to_idx and ind_code in industry_to_idx:
        print(f"\n📊 Creating waterfall plot for {title}...")
        
        c_idx = country_to_idx[country_code]
        i_idx = industry_to_idx[ind_code]
        
        # Get features
        country_feats = country_features_scaled[c_idx:c_idx+1]
        industry_vars = industry_variables_scaled[i_idx:i_idx+1]
        combined_feats = np.concatenate([country_feats[0], industry_vars[0]])
        
        # Get prediction
        pred = predict_with_features(c_idx, i_idx)
        
        # Create explainer for this specific prediction
        def single_predict(X):
            return global_predict(X)
        
        explainer = shap.KernelExplainer(single_predict, background_global)
        shap_values_single = explainer.shap_values(combined_feats.reshape(1, -1), nsamples=50)
        
        # Waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=explainer.expected_value,
                data=combined_feats,
                feature_names=feature_names
            ),
            show=False,
            max_display=15
        )
        plt.title(f'Waterfall Plot: {title}\nPredicted: {pred:.2f}')
        plt.tight_layout()
        plt.savefig(f'../results/plots/shap/waterfall_{country_code}_{ind_code}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# SAVE SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE - SUMMARY")
print("=" * 60)
print("\nWhat you've learned:")
print("  • Industry Variables SHAP: How your 8 scores affect predictions")
print("  • Country Features SHAP: How economic indicators affect predictions")
print("  • Global Importance: Most important features across all predictions")
print("  • Waterfall Plots: Detailed breakdown for specific predictions")
print("\nResults saved to:")
print("  • ../results/plots/shap/ - All SHAP visualizations")
print("=" * 60)