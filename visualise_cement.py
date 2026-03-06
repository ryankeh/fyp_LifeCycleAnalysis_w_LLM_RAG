import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the CSV file
df = pd.read_csv('combined_carbon_intensity_dataset.csv')

# Filter for industry 327310 (Cement manufacturing)
industry_code = '327310'
industry_data = df[df['industry_code'].astype(str) == industry_code].copy()

# Check if we have data
if len(industry_data) == 0:
    print(f"No data found for industry code {industry_code}")
    exit()

# Sort by carbon intensity
industry_data = industry_data.sort_values('carbon_intensity', ascending=False)

# Set up the visualization style
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure with 4 subplots (2x2 grid)
fig = plt.figure(figsize=(20, 16))

# 1. Main bar chart - Carbon intensity by country (Top Left)
ax1 = plt.subplot(2, 2, 1)
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(industry_data)))
bars = ax1.barh(range(len(industry_data)), industry_data['carbon_intensity'].values, color=colors)
ax1.set_yticks(range(len(industry_data)))
ax1.set_yticklabels(industry_data['Country'].values, fontsize=9)
ax1.set_xlabel('Carbon Intensity', fontsize=11)
ax1.set_title(f'Cement Manufacturing (327310)\nCarbon Intensity by Country', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# Add value labels
for i, v in enumerate(industry_data['carbon_intensity']):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)

# Add vertical line for global average
global_avg = industry_data['carbon_intensity'].mean()
ax1.axvline(global_avg, color='red', linestyle='--', linewidth=2, label=f'Global Avg: {global_avg:.3f}')
ax1.legend(loc='lower right', fontsize=9)

# 2. Distribution histogram (Top Right)
ax2 = plt.subplot(2, 2, 2)
sns.histplot(data=industry_data, x='carbon_intensity', bins=min(15, len(industry_data)), 
             kde=True, ax=ax2, color='steelblue')
ax2.axvline(global_avg, color='red', linestyle='--', linewidth=2, label=f'Mean: {global_avg:.3f}')
ax2.axvline(industry_data['carbon_intensity'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {industry_data["carbon_intensity"].median():.3f}')
ax2.set_xlabel('Carbon Intensity', fontsize=11)
ax2.set_ylabel('Number of Countries', fontsize=11)
ax2.set_title('Distribution Across Countries', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# 3. Scatter plot (Bottom Left) - Previously 4th figure
ax3 = plt.subplot(2, 2, 3)
scatter = ax3.scatter(industry_data['gdp_per_capita_ppp'].fillna(0), 
                     industry_data['carbon_intensity'], 
                     c=industry_data['coal_electricity_pct'].fillna(0), 
                     s=100, cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('GDP per Capita (PPP)', fontsize=11)
ax3.set_ylabel('Carbon Intensity', fontsize=11)
ax3.set_title('Carbon Intensity vs GDP per Capita\n(Colored by Coal Electricity %)', 
              fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Coal Electricity %', rotation=270, labelpad=15)

# Add country labels for extreme points
for idx, row in industry_data.iterrows():
    if row['carbon_intensity'] > industry_data['carbon_intensity'].quantile(0.9) or \
       row['carbon_intensity'] < industry_data['carbon_intensity'].quantile(0.1):
        ax3.annotate(row['Country'], (row['gdp_per_capita_ppp'], row['carbon_intensity']), 
                    fontsize=8, alpha=0.7)

# 4. Summary statistics (Bottom Right) - Previously 6th figure
ax4 = plt.subplot(2, 2, 4)
ax4.axis('tight')
ax4.axis('off')

# Prepare top and bottom countries text with proper formatting
top1_country = industry_data.iloc[0]['Country'] if len(industry_data) > 0 else 'N/A'
top1_value = f"{industry_data.iloc[0]['carbon_intensity']:.4f}" if len(industry_data) > 0 else 'N/A'

top2_country = industry_data.iloc[1]['Country'] if len(industry_data) > 1 else 'N/A'
top2_value = f"{industry_data.iloc[1]['carbon_intensity']:.4f}" if len(industry_data) > 1 else 'N/A'

top3_country = industry_data.iloc[2]['Country'] if len(industry_data) > 2 else 'N/A'
top3_value = f"{industry_data.iloc[2]['carbon_intensity']:.4f}" if len(industry_data) > 2 else 'N/A'

bottom1_country = industry_data.iloc[-1]['Country'] if len(industry_data) > 0 else 'N/A'
bottom1_value = f"{industry_data.iloc[-1]['carbon_intensity']:.4f}" if len(industry_data) > 0 else 'N/A'

bottom2_country = industry_data.iloc[-2]['Country'] if len(industry_data) > 1 else 'N/A'
bottom2_value = f"{industry_data.iloc[-2]['carbon_intensity']:.4f}" if len(industry_data) > 1 else 'N/A'

bottom3_country = industry_data.iloc[-3]['Country'] if len(industry_data) > 2 else 'N/A'
bottom3_value = f"{industry_data.iloc[-3]['carbon_intensity']:.4f}" if len(industry_data) > 2 else 'N/A'

# Summary statistics text
stats_text = f"""
CEMENT INDUSTRY SUMMARY (327310)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Countries with data: {len(industry_data)}

Global Statistics:
  Mean: {global_avg:.4f}
  Median: {industry_data['carbon_intensity'].median():.4f}
  Std Dev: {industry_data['carbon_intensity'].std():.4f}
  Min: {industry_data['carbon_intensity'].min():.4f}
  Max: {industry_data['carbon_intensity'].max():.4f}
  Range: {industry_data['carbon_intensity'].max() - industry_data['carbon_intensity'].min():.4f}

Top 3 Countries:
  1. {top1_country}: {top1_value}
  2. {top2_country}: {top2_value}
  3. {top3_country}: {top3_value}

Bottom 3 Countries:
  1. {bottom1_country}: {bottom1_value}
  2. {bottom2_country}: {bottom2_value}
  3. {bottom3_country}: {bottom3_value}

Correlations with Carbon Intensity:
"""

# Add correlations to the summary
correlation_text = ""
for col in ['gdp_per_capita_ppp', 'coal_electricity_pct', 'renewable_energy_pct', 
            'energy_intensity_level', 'urban_population_pct']:
    if col in industry_data.columns:
        valid_data = industry_data[['carbon_intensity', col]].dropna()
        if len(valid_data) > 1:
            corr = valid_data['carbon_intensity'].corr(valid_data[col])
            col_name = col.replace('_', ' ').title()
            correlation_text += f"  {col_name:25s}: {corr:+.4f}\n"

stats_text += correlation_text

ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Cement Manufacturing Industry Analysis (Code: {industry_code})\nCarbon Intensity Across Countries', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cement_industry_327310_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed data in console
print("\n" + "="*100)
print(f"CEMENT MANUFACTURING INDUSTRY (327310) - DETAILED DATA")
print("="*100)
print(f"\nNumber of countries with data: {len(industry_data)}")

print("\n" + "="*100)
print("TOP 10 COUNTRIES - DETAILED DATA")
print("="*100)
display_cols = ['Country', 'carbon_intensity', 'gdp_per_capita_ppp', 'coal_electricity_pct', 
                'renewable_energy_pct', 'energy_intensity_level']
print(industry_data[display_cols].head(10).to_string(index=False))

print("\n" + "="*100)
print("BOTTOM 10 COUNTRIES - DETAILED DATA")
print("="*100)
print(industry_data[display_cols].tail(10).to_string(index=False))