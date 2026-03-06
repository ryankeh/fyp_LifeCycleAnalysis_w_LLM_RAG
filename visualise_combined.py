import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('combined_carbon_intensity_dataset.csv')

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

# Calculate missing values
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
complete_percentage = 100 - missing_percentage

# Create a DataFrame with missing value statistics
missing_df = pd.DataFrame({
    'Variable': missing_count.index,
    'Missing_Count': missing_count.values,
    'Missing_Percentage': missing_percentage.values,
    'Complete_Percentage': complete_percentage.values
})

# Sort by missing percentage in descending order
missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
print("\nMissing Values Statistics:")
print(missing_df)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Bar plot of missing percentages
ax1 = axes[0, 0]
bars = ax1.barh(missing_df['Variable'], missing_df['Missing_Percentage'], color='skyblue')
ax1.set_xlabel('Missing Percentage (%)')
ax1.set_title('Missing Values by Variable')
ax1.axvline(x=0, color='black', linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, missing_df['Missing_Percentage'])):
    if val > 0:
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center')

# 2. Pie chart of overall missing vs complete
ax2 = axes[0, 1]
total_missing = df.isnull().sum().sum()
total_cells = df.size
missing_total_pct = (total_missing / total_cells) * 100
complete_total_pct = 100 - missing_total_pct

ax2.pie([complete_total_pct, missing_total_pct], 
        labels=['Complete', 'Missing'],
        autopct='%1.1f%%',
        colors=['lightgreen', 'salmon'],
        explode=(0, 0.1))
ax2.set_title(f'Overall Data Completeness\nTotal Cells: {total_cells:,}')

# 3. Heatmap of missing values
ax3 = axes[1, 0]
# Create a binary matrix of missing values (1 for missing, 0 for present)
missing_matrix = df.isnull().astype(int)
# Sample if dataset is too large for visualization
if len(df) > 1000:
    sample_size = 1000
    missing_matrix = missing_matrix.sample(n=sample_size, random_state=42)
    ax3.set_title(f'Missing Values Heatmap (Sample of {sample_size} rows)')
else:
    ax3.set_title('Missing Values Heatmap')

sns.heatmap(missing_matrix.T, cbar_kws={'label': 'Missing (1) / Present (0)'},
            cmap=['lightgreen', 'darkred'], ax=ax3)
ax3.set_xlabel('Row Index')
ax3.set_ylabel('Variables')

# 4. Bar plot of variables with missing values (top 20 if many)
ax4 = axes[1, 1]
missing_with_data = missing_df[missing_df['Missing_Percentage'] > 0]
if len(missing_with_data) > 0:
    ax4.bar(missing_with_data['Variable'][:20] if len(missing_with_data) > 20 
            else missing_with_data['Variable'], 
            missing_with_data['Missing_Percentage'][:20] if len(missing_with_data) > 20 
            else missing_with_data['Missing_Percentage'],
            color='lightcoral')
    ax4.set_ylabel('Missing Percentage (%)')
    ax4.set_title('Variables with Missing Values')
    ax4.tick_params(axis='x', rotation=45)
else:
    ax4.text(0.5, 0.5, 'No missing values found!', 
             ha='center', va='center', fontsize=14)
    ax4.set_title('Missing Values Status')

plt.tight_layout()
plt.show()

# Detailed statistics for each variable
print("\n" + "="*60)
print("DETAILED MISSING VALUE ANALYSIS")
print("="*60)

for idx, row in missing_df.iterrows():
    if row['Missing_Percentage'] > 0:
        print(f"\n{row['Variable']}:")
        print(f"  - Missing: {row['Missing_Count']:,} values ({row['Missing_Percentage']:.2f}%)")
        print(f"  - Complete: {row['Complete_Percentage']:.2f}%")
        
        # Show data type and some sample values
        dtype = df[row['Variable']].dtype
        print(f"  - Data type: {dtype}")
        
        # Show sample of non-missing values
        non_missing = df[row['Variable']].dropna()
        if len(non_missing) > 0:
            print(f"  - Sample values: {non_missing.iloc[:3].tolist()}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")
print(f"Variables with missing values: {(missing_df['Missing_Percentage'] > 0).sum()}")
print(f"Variables with no missing values: {(missing_df['Missing_Percentage'] == 0).sum()}")
print(f"Total missing cells: {total_missing:,} out of {total_cells:,} ({missing_total_pct:.2f}%)")