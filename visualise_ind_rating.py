# visualize_ratings.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('industry_ratings_output.csv')

# Get rating columns (those ending with '_score')
rating_cols = [col for col in df.columns if col.endswith('_score')]

# Set up the plot - 2x4 grid for 8 variables
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

# Create histogram for each variable
for i, col in enumerate(rating_cols):
    # Clean up variable name for display
    var_name = col.replace('_score', '').replace('_', ' ').title()
    
    # Plot histogram with bins 1-7
    axes[i].hist(df[col].dropna(), bins=[1,2,3,4,5,6,7,8], align='left', 
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_xticks([1,2,3,4,5,6,7])
    axes[i].set_xlabel('Rating (1-7 scale)')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{var_name}', fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    
    # Add count labels on bars
    counts, bins, patches = axes[i].hist(df[col].dropna(), bins=[1,2,3,4,5,6,7,8], 
                                         align='left', color='steelblue', edgecolor='black', alpha=0.7)
    for j, count in enumerate(counts):
        if count > 0:
            axes[i].text(j+1.2, count + 0.5, str(int(count)), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('rating_distributions_1to7.png', dpi=150)
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("📊 SUMMARY STATISTICS (1-7 Scale)")
print("="*60)
stats_df = df[rating_cols].describe().round(2)
print(stats_df)

# Print distribution summary
print("\n" + "="*60)
print("📈 DISTRIBUTION SUMMARY")
print("="*60)
for col in rating_cols:
    var_name = col.replace('_score', '').replace('_', ' ').title()
    value_counts = df[col].value_counts().sort_index()
    print(f"\n{var_name}:")
    for rating in range(1, 8):
        count = value_counts.get(rating, 0)
        percentage = (count / len(df)) * 100
        print(f"  Rating {rating}: {count} industries ({percentage:.1f}%)")

# Optional: Box plot to see distribution across variables
plt.figure(figsize=(14, 6))
df_melted = df[rating_cols].melt(var_name='Variable', value_name='Rating')
df_melted['Variable'] = df_melted['Variable'].str.replace('_score', '').str.replace('_', ' ').str.title()
sns.boxplot(data=df_melted, x='Variable', y='Rating')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Ratings Across Variables (1-7 Scale)', fontweight='bold')
plt.ylim(0.5, 7.5)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('rating_boxplots_1to7.png', dpi=150)
plt.show()