import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the Excel file
df = pd.read_csv('combined_carbon_intensity_dataset.csv')

# Calculate means across countries (for each country, average across all industries)
country_means = df.groupby('Country')['carbon_intensity'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).reset_index()
country_means.columns = ['Country', 'mean_intensity', 'std_intensity', 'industry_count']

# Calculate means across industries (for each industry, average across all countries)
industry_means = df.groupby('industry_name')['carbon_intensity'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False).reset_index()
industry_means.columns = ['Industry', 'mean_intensity', 'std_intensity', 'country_count']

# Set up the visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Top 20 Countries by Mean Carbon Intensity
ax1 = plt.subplot(2, 3, 1)
top20_countries = country_means.head(20)
bars1 = ax1.barh(range(len(top20_countries)), top20_countries['mean_intensity'].values)
ax1.set_yticks(range(len(top20_countries)))
ax1.set_yticklabels(top20_countries['Country'].values, fontsize=8)
ax1.set_xlabel('Mean Carbon Intensity', fontsize=10)
ax1.set_title('Top 20 Countries by Mean Carbon Intensity', fontsize=12, fontweight='bold')
ax1.invert_yaxis()  # Highest at top

# Add value labels
for i, (v, std) in enumerate(zip(top20_countries['mean_intensity'], top20_countries['std_intensity'])):
    ax1.text(v + 0.02, i, f'{v:.3f} (±{std:.3f})', va='center', fontsize=7)

# 2. Bottom 20 Countries by Mean Carbon Intensity
ax2 = plt.subplot(2, 3, 2)
bottom20_countries = country_means.tail(20)
bars2 = ax2.barh(range(len(bottom20_countries)), bottom20_countries['mean_intensity'].values)
ax2.set_yticks(range(len(bottom20_countries)))
ax2.set_yticklabels(bottom20_countries['Country'].values, fontsize=8)
ax2.set_xlabel('Mean Carbon Intensity', fontsize=10)
ax2.set_title('Bottom 20 Countries by Mean Carbon Intensity', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

# Add value labels
for i, (v, std) in enumerate(zip(bottom20_countries['mean_intensity'], bottom20_countries['std_intensity'])):
    ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=7)

# 3. Top 20 Industries by Mean Carbon Intensity
ax3 = plt.subplot(2, 3, 3)
top20_industries = industry_means.head(20)
bars3 = ax3.barh(range(len(top20_industries)), top20_industries['mean_intensity'].values)
ax3.set_yticks(range(len(top20_industries)))
ax3.set_yticklabels([ind[:30] + '...' if len(ind) > 30 else ind for ind in top20_industries['Industry'].values], fontsize=7)
ax3.set_xlabel('Mean Carbon Intensity', fontsize=10)
ax3.set_title('Top 20 Industries by Mean Carbon Intensity', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Add value labels
for i, (v, std) in enumerate(zip(top20_industries['mean_intensity'], top20_industries['std_intensity'])):
    ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=7)

# 4. Distribution of Country Means
ax4 = plt.subplot(2, 3, 4)
sns.histplot(data=country_means, x='mean_intensity', bins=30, kde=True, ax=ax4)
ax4.axvline(country_means['mean_intensity'].mean(), color='red', linestyle='--', label=f"Mean: {country_means['mean_intensity'].mean():.3f}")
ax4.axvline(country_means['mean_intensity'].median(), color='green', linestyle='--', label=f"Median: {country_means['mean_intensity'].median():.3f}")
ax4.set_xlabel('Mean Carbon Intensity', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Distribution of Country Mean Intensities', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)

# 5. Distribution of Industry Means
ax5 = plt.subplot(2, 3, 5)
sns.histplot(data=industry_means, x='mean_intensity', bins=30, kde=True, ax=ax5)
ax5.axvline(industry_means['mean_intensity'].mean(), color='red', linestyle='--', label=f"Mean: {industry_means['mean_intensity'].mean():.3f}")
ax5.axvline(industry_means['mean_intensity'].median(), color='green', linestyle='--', label=f"Median: {industry_means['mean_intensity'].median():.3f}")
ax5.set_xlabel('Mean Carbon Intensity', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Distribution of Industry Mean Intensities', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)

# 6. Summary Statistics Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Prepare summary statistics
summary_data = [
    ['Metric', 'Countries', 'Industries'],
    ['Count', f"{len(country_means)}", f"{len(industry_means)}"],
    ['Mean', f"{country_means['mean_intensity'].mean():.4f}", f"{industry_means['mean_intensity'].mean():.4f}"],
    ['Std Dev', f"{country_means['mean_intensity'].std():.4f}", f"{industry_means['mean_intensity'].std():.4f}"],
    ['Min', f"{country_means['mean_intensity'].min():.4f}", f"{industry_means['mean_intensity'].min():.4f}"],
    ['25%', f"{country_means['mean_intensity'].quantile(0.25):.4f}", f"{industry_means['mean_intensity'].quantile(0.25):.4f}"],
    ['50% (Median)', f"{country_means['mean_intensity'].median():.4f}", f"{industry_means['mean_intensity'].median():.4f}"],
    ['75%', f"{country_means['mean_intensity'].quantile(0.75):.4f}", f"{industry_means['mean_intensity'].quantile(0.75):.4f}"],
    ['Max', f"{country_means['mean_intensity'].max():.4f}", f"{industry_means['mean_intensity'].max():.4f}"],
]

table = ax6.table(cellText=summary_data, loc='center', cellLoc='center', colWidths=[0.2, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Highlight header row
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white')

plt.suptitle('Carbon Emission Intensity Analysis: Countries vs Industries', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('carbon_intensity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print top 10 highest values for both
print("\n" + "="*80)
print("TOP 10 HIGHEST COUNTRY MEAN CARBON INTENSITIES")
print("="*80)
print(country_means[['Country', 'mean_intensity', 'industry_count']].head(10).to_string(index=False))

print("\n" + "="*80)
print("TOP 10 HIGHEST INDUSTRY MEAN CARBON INTENSITIES")
print("="*80)
print(industry_means[['Industry', 'mean_intensity', 'country_count']].head(10).to_string(index=False))

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Number of unique countries: {len(country_means)}")
print(f"Number of unique industries: {len(industry_means)}")
print(f"Total data points: {len(df)}")
print(f"\nGlobal mean carbon intensity: {df['carbon_intensity'].mean():.4f}")
print(f"Global std carbon intensity: {df['carbon_intensity'].std():.4f}")