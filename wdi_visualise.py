import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the values file
df = pd.read_excel('carbon_intensity_indicators_values.xlsx', sheet_name='Indicator Values')

# Define the 8 indicators with shorter names for plotting
indicators_short = {
    'GDP per capita, PPP (constant 2021 international $)': 'GDP per Capita (PPP)',
    'Industry (including construction), value added (% of GDP)': 'Industry (% of GDP)',
    'Renewable energy consumption (% of total final energy consumption)': 'Renewable Energy (%)',
    'Electricity production from coal sources (% of total)': 'Coal Electricity (%)',
    'Energy intensity level of primary energy (MJ/$2021 PPP GDP)': 'Energy Intensity',
    'GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)': 'GDP per Energy Unit',
    'Urban population (% of total population)': 'Urban Population (%)',
    'Total natural resources rents (% of GDP)': 'Natural Resources Rents (%)'
}

# Remove aggregate regions and keep only countries
# (you can adjust this list based on what you consider "countries")
aggregate_indicators = ['World', 'East Asia & Pacific', 'Europe & Central Asia', 
                        'Latin America & Caribbean', 'Middle East & North Africa', 
                        'North America', 'South Asia', 'Sub-Saharan Africa',
                        'European Union', 'OECD members', 'IDA & IBRD total',
                        'Low income', 'Middle income', 'High income',
                        'East Asia & Pacific (IDA & IBRD countries)',
                        'Europe & Central Asia (IDA & IBRD countries)',
                        'Latin America & Caribbean (IDA & IBRD countries)',
                        'Middle East, North Africa, Afghanistan & Pakistan',
                        'Sub-Saharan Africa (IDA & IBRD countries)',
                        'Arab World', 'Africa Eastern and Southern', 'Africa Western and Central',
                        'Central Europe and the Baltics', 'Caribbean small states',
                        'Pacific island small states', 'Small states',
                        'Early-demographic dividend', 'Late-demographic dividend',
                        'Pre-demographic dividend', 'Post-demographic dividend',
                        'Fragile and conflict affected situations',
                        'Heavily indebted poor countries (HIPC)',
                        'Least developed countries: UN classification']

# Filter out aggregate regions
df_countries = df[~df['Country Name'].isin(aggregate_indicators)].copy()
print(f"Number of countries after filtering: {len(df_countries)}")

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create a figure with subplots (4 rows, 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

# Plot each indicator
for idx, (indicator_long, indicator_short) in enumerate(indicators_short.items()):
    ax = axes[idx]
    
    # Get data, drop NaN
    data = df_countries[indicator_long].dropna()
    
    # Create histogram with KDE
    ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Add labels and title
    ax.set_xlabel(indicator_short, fontsize=11)
    ax.set_ylabel('Number of Countries', fontsize=11)
    ax.set_title(f'Distribution of {indicator_short}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add text with number of countries
    ax.text(0.98, 0.95, f'n = {len(data)}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('indicator_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a second figure with boxplots to show outliers better
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10))
axes2 = axes2.flatten()

for idx, (indicator_long, indicator_short) in enumerate(indicators_short.items()):
    ax = axes2[idx]
    
    # Get data, drop NaN
    data = df_countries[indicator_long].dropna()
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, showfliers=True, flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Add statistics
    ax.text(1.1, 0.95, f'Mean: {data.mean():.1f}\nMedian: {data.median():.1f}\nSD: {data.std():.1f}', 
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(indicator_short, fontsize=11, fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])

plt.tight_layout()
plt.savefig('indicator_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(12, 10))

# Calculate correlation matrix
corr_data = df_countries[list(indicators_short.keys())].copy()
corr_data.columns = list(indicators_short.values())
corr_matrix = corr_data.corr()

# Create heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            fmt='.2f', annot_kws={'size': 9})

plt.title('Correlation Matrix of Carbon Intensity Indicators', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('indicator_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR ALL COUNTRIES")
print("="*60)

summary_stats = []
for indicator_long, indicator_short in indicators_short.items():
    data = df_countries[indicator_long].dropna()
    summary_stats.append({
        'Indicator': indicator_short,
        'Count': len(data),
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Skewness': data.skew()
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

# Identify countries with extreme values for each indicator
print("\n" + "="*60)
print("COUNTRIES WITH EXTREME VALUES")
print("="*60)

for indicator_long, indicator_short in indicators_short.items():
    data = df_countries[indicator_long].dropna()
    
    # Top 5 highest
    top5 = df_countries.loc[data.nlargest(5).index, ['Country Name', indicator_long]]
    
    # Bottom 5 lowest (excluding zeros if many structural zeros)
    if indicator_long == 'Electricity production from coal sources (% of total)':
        # For coal, show countries with highest values only
        print(f"\n{indicator_short} - Top 5 (highest coal dependence):")
        for _, row in top5.iterrows():
            print(f"  {row['Country Name']}: {row[indicator_long]:.1f}%")
    else:
        bottom5 = df_countries.loc[data.nsmallest(5).index, ['Country Name', indicator_long]]
        
        print(f"\n{indicator_short}:")
        print("  Top 5 highest:")
        for _, row in top5.iterrows():
            print(f"    {row['Country Name']}: {row[indicator_long]:.2f}")
        print("  Top 5 lowest:")
        for _, row in bottom5.iterrows():
            print(f"    {row['Country Name']}: {row[indicator_long]:.2f}")