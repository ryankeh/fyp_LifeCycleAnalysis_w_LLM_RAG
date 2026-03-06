"""
Script 8: Model Comparison Visualization
Creates comprehensive graphs comparing all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create directories
Path("../results/plots/comparison").mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# MODEL RESULTS DATA
# ============================================================================

models = ['Neural Network', 'Random Forest', 'XGBoost', 'LightGBM']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']

results = {
    'Neural Network': {
        'MAE': 0.223404,
        'RMSE': 0.514925,
        'R2': 0.713091,
        'MAPE': 49.32,
        'SMAPE': 35.94,
        'MedAE': 0.130833
    },
    'Random Forest': {
        'MAE': 0.208356,
        'RMSE': 0.562355,
        'R2': 0.657802,
        'MAPE': 41.32,
        'SMAPE': 31.12,
        'MedAE': 0.105991
    },
    'XGBoost': {
        'MAE': 0.200478,
        'RMSE': 0.505363,
        'R2': 0.723647,
        'MAPE': 39.80,
        'SMAPE': 31.33,
        'MedAE': 0.111545
    },
    'LightGBM': {
        'MAE': 0.211398,
        'RMSE': 0.542451,
        'R2': 0.681596,
        'MAPE': 47.33,
        'SMAPE': 33.94,
        'MedAE': 0.123618
    }
}

# Create DataFrame for easy plotting
df = pd.DataFrame(results).T
print("Model Comparison Data:")
print(df.round(4))

# ============================================================================
# FIGURE 1: Bar Chart Comparison of All Metrics
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold')

metrics = ['MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE', 'MedAE']
titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 
          'R² Score', 'Mean Absolute Percentage Error (MAPE %)', 
          'Symmetric MAPE (SMAPE %)', 'Median Absolute Error (MedAE)']
ylabels = ['Error Value', 'Error Value', 'Score', 'Percentage (%)', 'Percentage (%)', 'Error Value']

for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[idx // 3, idx % 3]
    
    values = [results[model][metric] for model in models]
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}' if metric not in ['MAPE', 'SMAPE'] else f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best value
    if metric == 'R2':
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('../results/plots/comparison/01_all_metrics_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 2: Radar Chart for Holistic Comparison
# ============================================================================

def create_radar_chart():
    # Normalize metrics (0-1 scale, where 1 is best)
    normalized = {}
    for model in models:
        normalized[model] = {}
        for metric in metrics:
            if metric == 'R2':
                # For R2, higher is better
                min_val = min(df[metric])
                max_val = max(df[metric])
                normalized[model][metric] = (df.loc[model, metric] - min_val) / (max_val - min_val) if max_val > min_val else 1
            else:
                # For error metrics, lower is better
                min_val = min(df[metric])
                max_val = max(df[metric])
                normalized[model][metric] = 1 - (df.loc[model, metric] - min_val) / (max_val - min_val) if max_val > min_val else 1
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Model Performance Radar Chart', fontsize=16, fontweight='bold')
    
    for i, model in enumerate(models):
        values = [normalized[model][m] for m in metrics]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('../results/plots/comparison/02_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

create_radar_chart()

# ============================================================================
# FIGURE 3: Grouped Bar Chart for Error Metrics
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.15
multiplier = 0

error_metrics = ['MAE', 'RMSE', 'MedAE']
for metric in error_metrics:
    offset = width * multiplier
    values = [results[model][metric] for model in models]
    bars = ax.bar(x + offset, values, width, label=metric, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    multiplier += 1

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Error Values', fontsize=12)
ax.set_title('Error Metrics Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/plots/comparison/03_error_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 4: Percentage Error Metrics
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.35

percentage_metrics = ['MAPE', 'SMAPE']
colors_pct = ['#FF9999', '#66B2FF']

for i, metric in enumerate(percentage_metrics):
    offset = width * i
    values = [results[model][metric] for model in models]
    bars = ax.bar(x + offset, values, width, label=metric, 
                  color=colors_pct[i], edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Percentage Error (%)', fontsize=12)
ax.set_title('Percentage Error Metrics Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width/2)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/plots/comparison/04_percentage_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 5: R² Score Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

r2_values = [results[model]['R2'] for model in models]
bars = ax.bar(models, r2_values, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best
best_idx = np.argmax(r2_values)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('R² Score Comparison (Higher is Better)', fontsize=14, fontweight='bold')
ax.set_ylim(0.6, 0.75)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/plots/comparison/05_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 6: Heatmap of All Metrics
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap data
heatmap_data = pd.DataFrame(index=models, columns=metrics)
for model in models:
    for metric in metrics:
        heatmap_data.loc[model, metric] = results[model][metric]

# Normalize for color scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
heatmap_normalized = pd.DataFrame(
    scaler.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns
)

# Create heatmap
sns.heatmap(heatmap_normalized, annot=heatmap_data.round(3), 
            fmt='.3f', cmap='RdYlGn_r' if 'R2' in heatmap_data.columns else 'RdYlGn',
            center=0, ax=ax, cbar_kws={'label': 'Standardized Score'})

ax.set_title('Model Performance Heatmap\n(Green = Better, Red = Worse)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/plots/comparison/06_performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 7: Spider Chart of Rankings
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate rankings (1 = best, 4 = worst)
rankings = {}
for metric in metrics:
    if metric == 'R2':
        # For R2, higher rank = better
        sorted_models = df[metric].sort_values(ascending=False).index
    else:
        # For error metrics, lower rank = better
        sorted_models = df[metric].sort_values().index
    
    for rank, model in enumerate(sorted_models, 1):
        if model not in rankings:
            rankings[model] = []
        rankings[model].append(rank)

# Create bar chart of average rankings
avg_rankings = {model: np.mean(ranks) for model, ranks in rankings.items()}
avg_rankings_df = pd.DataFrame.from_dict(avg_rankings, orient='index', columns=['Average Rank'])
avg_rankings_df = avg_rankings_df.sort_values('Average Rank')

colors_rank = ['#FFD700', '#C0C0C0', '#CD7F32', '#A9A9A9']  # Gold, Silver, Bronze, Grey

bars = ax.bar(avg_rankings_df.index, avg_rankings_df['Average Rank'], 
              color=colors_rank[:len(avg_rankings_df)], edgecolor='black', linewidth=1)

# Add value labels
for bar, val in zip(bars, avg_rankings_df['Average Rank']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Average Rank (Lower is Better)', fontsize=12)
ax.set_title('Model Rankings Summary\n(1 = Best, 4 = Worst)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 4.5)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Add rank labels
ax.text(0.02, 0.98, '🥇 Best', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.02, 0.93, '🥈 2nd', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.02, 0.88, '🥉 3rd', transform=ax.transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.savefig('../results/plots/comparison/07_rankings_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 8: Trade-off Plot (MAE vs RMSE)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

for i, model in enumerate(models):
    ax.scatter(results[model]['MAE'], results[model]['RMSE'], 
              s=200, c=[colors[i]], marker=markers[i], label=model, 
              edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add model name as annotation
    ax.annotate(model, (results[model]['MAE'], results[model]['RMSE']),
               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Add quadrant lines
ax.axhline(y=np.mean([r['RMSE'] for r in results.values()]), color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=np.mean([r['MAE'] for r in results.values()]), color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('MAE (Lower is Better)', fontsize=12)
ax.set_ylabel('RMSE (Lower is Better)', fontsize=12)
ax.set_title('MAE vs RMSE Trade-off\n(Bottom-Left is Best)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('../results/plots/comparison/08_mae_vs_rmse.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 9: Percentage Improvement Over Baseline
# ============================================================================

# Use Neural Network as baseline
baseline_model = 'Neural Network'
improvements = {}

for model in models:
    if model != baseline_model:
        improvements[model] = {}
        for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'MedAE']:  # Error metrics
            baseline_val = results[baseline_model][metric]
            model_val = results[model][metric]
            pct_improvement = ((baseline_val - model_val) / baseline_val) * 100
            improvements[model][metric] = pct_improvement

# Create improvement plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(list(improvements.keys())))
width = 0.15
multiplier = 0

for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'MedAE']:
    offset = width * multiplier
    values = [improvements[model][metric] for model in improvements.keys()]
    bars = ax.bar(x + offset, values, width, label=metric, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        color = 'green' if val > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=8, color=color, fontweight='bold')
    
    multiplier += 1

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Improvement over Neural Network (%)', fontsize=12)
ax.set_title(f'Percentage Improvement Over {baseline_model}\n(Positive = Better, Negative = Worse)', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(improvements.keys(), rotation=45, ha='right')
ax.legend(loc='upper right', ncol=3)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/plots/comparison/09_improvement_over_baseline.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print("\nPerformance Rankings (1=best, 4=worst):")
print("-"*40)

ranking_summary = pd.DataFrame(rankings, index=metrics).T
ranking_summary['Average Rank'] = ranking_summary.mean(axis=1)
ranking_summary = ranking_summary.sort_values('Average Rank')

print(ranking_summary.round(2))

print("\n" + "="*60)
print(f"🏆 Overall Winner: {ranking_summary.index[0]} (Avg Rank: {ranking_summary.iloc[0, -1]:.2f})")
print(f"🥈 Runner Up: {ranking_summary.index[1]} (Avg Rank: {ranking_summary.iloc[1, -1]:.2f})")
print(f"🥉 Third Place: {ranking_summary.index[2]} (Avg Rank: {ranking_summary.iloc[2, -1]:.2f})")
print("="*60)
print(f"\nAll comparison plots saved to: ../results/plots/comparison/")