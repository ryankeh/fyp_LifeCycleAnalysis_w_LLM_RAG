# ============================================================================
# RELATIVE ERROR ANALYSIS
# ============================================================================
import pandas as pd
import numpy as np
import os

# Load your evaluation metrics if already saved
# eval_df = pd.read_excel('hotdeck_results/evaluation_metrics_all_columns.xlsx')

# Or recalculate with relative error from your existing data
def calculate_relative_error(original_df, imputed_df, masked_df, output_dir='hotdeck_results'):
    """
    Calculate comprehensive relative error metrics for imputed values.
    """
    
    print("="*70)
    print("RELATIVE ERROR ANALYSIS")
    print("="*70)
    
    # Get all columns that were imputed
    imputed_columns = []
    for col in original_df.columns:
        if col not in ['Country', 'Country Code'] and pd.api.types.is_numeric_dtype(original_df[col]):
            if col in masked_df.columns and col in imputed_df.columns:
                if masked_df[col].isna().sum() > 0:
                    imputed_columns.append(col)
    
    print(f"Analyzing relative error for {len(imputed_columns)} columns...")
    
    results = []
    
    for col in imputed_columns:
        # Find imputed positions
        imputed_mask = masked_df[col].isna() & imputed_df[col].notna()
        
        if imputed_mask.sum() == 0:
            continue
        
        # Get true and imputed values
        true_vals = []
        imputed_vals = []
        
        for idx in imputed_df[imputed_mask].index:
            country_code = imputed_df.loc[idx, 'Country Code']
            
            true_match = original_df[original_df['Country Code'] == country_code]
            imputed_match = imputed_df[imputed_df['Country Code'] == country_code]
            
            if len(true_match) > 0 and len(imputed_match) > 0:
                true_val = true_match[col].values[0]
                imputed_val = imputed_match[col].values[0]
                
                if pd.notna(true_val) and pd.notna(imputed_val):
                    true_vals.append(true_val)
                    imputed_vals.append(imputed_val)
        
        if len(true_vals) == 0:
            continue
        
        true_vals = np.array(true_vals)
        imputed_vals = np.array(imputed_vals)
        
        # Calculate absolute error
        abs_errors = np.abs(imputed_vals - true_vals)
        
        # ============ RELATIVE ERROR METRICS ============
        
        # 1. Mean Absolute Percentage Error (MAPE) - most common
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((true_vals - imputed_vals) / true_vals)) * 100
        
        # 2. Median Absolute Percentage Error (MdAPE) - robust to outliers
        with np.errstate(divide='ignore', invalid='ignore'):
            mdape = np.median(np.abs((true_vals - imputed_vals) / true_vals)) * 100
        
        # 3. Relative MAE (normalized by mean)
        rel_mae = np.mean(abs_errors) / np.mean(true_vals) * 100 if np.mean(true_vals) != 0 else np.nan
        
        # 4. Relative RMSE (normalized by mean)
        rmse = np.sqrt(np.mean((imputed_vals - true_vals) ** 2))
        rel_rmse = rmse / np.mean(true_vals) * 100 if np.mean(true_vals) != 0 else np.nan
        
        # 5. Symmetric MAPE (sMAPE) - handles zeros better
        with np.errstate(divide='ignore', invalid='ignore'):
            smape = np.mean(2 * np.abs(imputed_vals - true_vals) / (np.abs(true_vals) + np.abs(imputed_vals))) * 100
        
        # 6. Percentage of values within error thresholds
        pct_within_10pct = np.mean(abs_errors <= (0.1 * np.abs(true_vals))) * 100
        pct_within_20pct = np.mean(abs_errors <= (0.2 * np.abs(true_vals))) * 100
        pct_within_50pct = np.mean(abs_errors <= (0.5 * np.abs(true_vals))) * 100
        
        # 7. Scale information
        mean_true = np.mean(true_vals)
        median_true = np.median(true_vals)
        std_true = np.std(true_vals)
        min_true = np.min(true_vals)
        max_true = np.max(true_vals)
        
        results.append({
            'column': col,
            'n_imputed': len(true_vals),
            
            # Scale info
            'mean_true': mean_true,
            'median_true': median_true,
            'std_true': std_true,
            'min_true': min_true,
            'max_true': max_true,
            
            # Absolute error
            'MAE': np.mean(abs_errors),
            'RMSE': rmse,
            
            # Relative error (%)
            'MAPE_%': mape,
            'MdAPE_%': mdape,
            'sMAPE_%': smape,
            'Rel_MAE_%': rel_mae,
            'Rel_RMSE_%': rel_rmse,
            
            # Accuracy thresholds (%)
            'within_10%': pct_within_10pct,
            'within_20%': pct_within_20pct,
            'within_50%': pct_within_50pct,
            
            # Error characterization
            'bias': np.mean(imputed_vals - true_vals),
            'bias_%': np.mean(imputed_vals - true_vals) / mean_true * 100 if mean_true != 0 else np.nan,
        })
    
    # Create DataFrame
    rel_df = pd.DataFrame(results)
    
    # Sort by MAPE (worst first for investigation)
    rel_df = rel_df.sort_values('MAPE_%', ascending=False)
    
    # Save results
    output_path = os.path.join(output_dir, 'relative_error_analysis.xlsx')
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Full results
        rel_df.to_excel(writer, sheet_name='All_Columns', index=False)
        
        # Summary statistics
        summary = pd.DataFrame({
            'Metric': ['MAPE (%)', 'MdAPE (%)', 'sMAPE (%)', 'Rel_MAE (%)', 'Rel_RMSE (%)', 
                      'Within 10% (%)', 'Within 20% (%)', 'Within 50% (%)'],
            'Mean': [
                rel_df['MAPE_%'].mean(),
                rel_df['MdAPE_%'].mean(),
                rel_df['sMAPE_%'].mean(),
                rel_df['Rel_MAE_%'].mean(),
                rel_df['Rel_RMSE_%'].mean(),
                rel_df['within_10%'].mean(),
                rel_df['within_20%'].mean(),
                rel_df['within_50%'].mean()
            ],
            'Median': [
                rel_df['MAPE_%'].median(),
                rel_df['MdAPE_%'].median(),
                rel_df['sMAPE_%'].median(),
                rel_df['Rel_MAE_%'].median(),
                rel_df['Rel_RMSE_%'].median(),
                rel_df['within_10%'].median(),
                rel_df['within_20%'].median(),
                rel_df['within_50%'].median()
            ],
            'Std': [
                rel_df['MAPE_%'].std(),
                rel_df['MdAPE_%'].std(),
                rel_df['sMAPE_%'].std(),
                rel_df['Rel_MAE_%'].std(),
                rel_df['Rel_RMSE_%'].std(),
                rel_df['within_10%'].std(),
                rel_df['within_20%'].std(),
                rel_df['within_50%'].std()
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Best performing by relative error
        best_rel = rel_df.nsmallest(20, 'MAPE_%')[['column', 'n_imputed', 'mean_true', 'MAE', 'MAPE_%', 'within_20%']]
        best_rel.to_excel(writer, sheet_name='Best_20_Relative', index=False)
        
        # Worst performing by relative error
        worst_rel = rel_df.nlargest(20, 'MAPE_%')[['column', 'n_imputed', 'mean_true', 'MAE', 'MAPE_%', 'within_20%']]
        worst_rel.to_excel(writer, sheet_name='Worst_20_Relative', index=False)
        
        # Columns with very small values (prone to high MAPE)
        small_values = rel_df[rel_df['mean_true'] < 0.1].sort_values('MAPE_%', ascending=False)
        if len(small_values) > 0:
            small_values.to_excel(writer, sheet_name='Small_Values_<0.1', index=False)
        
        # Columns with very large values
        large_values = rel_df[rel_df['mean_true'] > 10].sort_values('MAPE_%', ascending=False)
        if len(large_values) > 0:
            large_values.to_excel(writer, sheet_name='Large_Values_>10', index=False)
    
    print(f"\n✓ Relative error analysis saved to: {output_path}")
    
    return rel_df


def analyze_worst_performers(rel_df, original_df, imputed_df, masked_df, top_n=10):
    """
    Deep dive into the worst performing columns to understand why.
    """
    print("\n" + "="*70)
    print("DEEP DIVE: WORST PERFORMING COLUMNS")
    print("="*70)
    
    # Get worst columns by MAPE
    worst_cols = rel_df.nlargest(top_n, 'MAPE_%')[['column', 'MAPE_%', 'mean_true', 'MAE', 'n_imputed']]
    
    for idx, row in worst_cols.iterrows():
        col = row['column']
        
        print(f"\n{idx+1}. Column: {col}")
        print(f"   MAPE: {row['MAPE_%']:.1f}%")
        print(f"   MAE: {row['MAE']:.4f}")
        print(f"   Mean true value: {row['mean_true']:.4f}")
        print(f"   Imputed count: {row['n_imputed']}")
        
        # Get actual values for this column
        true_vals = original_df[col].dropna()
        imputed_mask = masked_df[col].isna() & imputed_df[col].notna()
        imputed_vals = imputed_df.loc[imputed_mask, col]
        
        print(f"\n   Value distribution:")
        print(f"     Original - min: {true_vals.min():.4f}, max: {true_vals.max():.4f}, median: {true_vals.median():.4f}")
        if len(imputed_vals) > 0:
            print(f"     Imputed  - min: {imputed_vals.min():.4f}, max: {imputed_vals.max():.4f}, median: {imputed_vals.median():.4f}")
        
        # Check if this is a zero/very small value problem
        if row['mean_true'] < 0.1:
            print(f"   ⚠️  Very small values (mean={row['mean_true']:.4f}) - MAPE is misleading")
        
        # Check if this is a sparse column
        non_null_pct = (true_vals.count() / len(original_df)) * 100
        print(f"   Data density: {non_null_pct:.1f}% of countries have non-null values")
        
        # Check if this column has outliers
        q1 = true_vals.quantile(0.25)
        q3 = true_vals.quantile(0.75)
        iqr = q3 - q1
        outliers = ((true_vals < (q1 - 1.5 * iqr)) | (true_vals > (q3 + 1.5 * iqr))).sum()
        print(f"   Outliers: {outliers} ({outliers/len(true_vals)*100:.1f}% of values)")
    
    return worst_cols


def visualize_relative_error(rel_df, output_dir='hotdeck_results'):
    """
    Create visualizations for relative error analysis.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Distribution of MAPE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Remove infinite values
    mape_clean = rel_df['MAPE_%'].replace([np.inf, -np.inf], np.nan).dropna()
    
    axes[0, 0].hist(mape_clean, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(mape_clean.median(), color='red', linestyle='--', label=f'Median: {mape_clean.median():.1f}%')
    axes[0, 0].axvline(mape_clean.mean(), color='green', linestyle='--', label=f'Mean: {mape_clean.mean():.1f}%')
    axes[0, 0].set_xlabel('MAPE (%)')
    axes[0, 0].set_ylabel('Number of Columns')
    axes[0, 0].set_title('Distribution of Mean Absolute Percentage Error')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. MAPE vs Mean Value (scale effect)
    axes[0, 1].scatter(rel_df['mean_true'], rel_df['MAPE_%'], alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Mean True Value')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].set_title('MAPE vs Scale of Values')
    axes[0, 1].set_xscale('log')
    axes[0, 1].axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% threshold')
    axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0, 1].legend()
    
    # 3. Accuracy thresholds
    thresholds_df = rel_df[['within_10%', 'within_20%', 'within_50%']].mean()
    axes[1, 0].bar(range(len(thresholds_df)), thresholds_df.values, tick_label=['±10%', '±20%', '±50%'])
    axes[1, 0].set_ylabel('Average % of Imputed Values')
    axes[1, 0].set_title('Accuracy Within Error Thresholds')
    axes[1, 0].set_ylim([0, 100])
    for i, v in enumerate(thresholds_df.values):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 4. Cumulative distribution of MAPE
    axes[1, 1].hist(mape_clean, bins=100, cumulative=True, density=True, 
                    histtype='step', linewidth=2)
    axes[1, 1].set_xlabel('MAPE (%)')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title('Cumulative Distribution of MAPE')
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='50th percentile')
    axes[1, 1].axhline(y=0.8, color='orange', linestyle='--', label='80th percentile')
    axes[1, 1].axhline(y=0.9, color='green', linestyle='--', label='90th percentile')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'relative_error_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Visualization saved to: {plot_path}")
    
    # Additional plot: MAE vs MAPE colored by scale
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create size based on mean value
    sizes = 20 + (rel_df['mean_true'] / rel_df['mean_true'].max()) * 100
    
    scatter = ax.scatter(rel_df['MAE'], rel_df['MAPE_%'], 
                        c=np.log10(rel_df['mean_true'] + 0.001), 
                        s=sizes, alpha=0.6, cmap='viridis')
    ax.set_xlabel('MAE (Absolute Error)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAE vs MAPE (color = log scale of mean value)')
    ax.set_xscale('log')
    plt.colorbar(scatter, label='log10(mean value)')
    
    # Add quadrant lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'mae_vs_mape_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Scatter plot saved to: {scatter_path}")


# ============================================================================
# RUN THE FULL RELATIVE ERROR ANALYSIS
# ============================================================================

def run_relative_error_analysis(original_df, imputed_df, masked_df, output_dir='hotdeck_results'):
    """
    Run complete relative error analysis pipeline.
    """
    
    # Step 1: Calculate relative error metrics
    rel_df = calculate_relative_error(original_df, imputed_df, masked_df, output_dir)
    
    # Step 2: Print summary statistics
    print("\n" + "="*70)
    print("RELATIVE ERROR SUMMARY")
    print("="*70)
    
    print(f"\nOverall MAPE: {rel_df['MAPE_%'].mean():.1f}%")
    print(f"Median MAPE: {rel_df['MAPE_%'].median():.1f}%")
    print(f"Overall sMAPE: {rel_df['sMAPE_%'].mean():.1f}%")
    
    print(f"\nAccuracy thresholds:")
    print(f"  Within 10%: {rel_df['within_10%'].mean():.1f}% of imputed values")
    print(f"  Within 20%: {rel_df['within_20%'].mean():.1f}% of imputed values")
    print(f"  Within 50%: {rel_df['within_50%'].mean():.1f}% of imputed values")
    
    # Step 3: Categorize columns by performance
    excellent = rel_df[rel_df['MAPE_%'] <= 10]
    good = rel_df[(rel_df['MAPE_%'] > 10) & (rel_df['MAPE_%'] <= 30)]
    fair = rel_df[(rel_df['MAPE_%'] > 30) & (rel_df['MAPE_%'] <= 50)]
    poor = rel_df[rel_df['MAPE_%'] > 50]
    
    print(f"\nPerformance categories:")
    print(f"  Excellent (≤10% error): {len(excellent)} columns")
    print(f"  Good (10-30% error): {len(good)} columns")
    print(f"  Fair (30-50% error): {len(fair)} columns")
    print(f"  Poor (>50% error): {len(poor)} columns")
    
    # Step 4: Check for small-value bias
    small_cols = rel_df[rel_df['mean_true'] < 0.1]
    if len(small_cols) > 0:
        small_poor = small_cols[small_cols['MAPE_%'] > 50]
        print(f"\n⚠️  {len(small_poor)} columns with mean < 0.1 have >50% MAPE")
        print("   This is likely due to division by very small numbers, not poor imputation")
    
    # Step 5: Analyze worst performers
    worst_cols = analyze_worst_performers(rel_df, original_df, imputed_df, masked_df, top_n=10)
    
    # Step 6: Create visualizations
    visualize_relative_error(rel_df, output_dir)
    
    # Step 7: Save worst performers for your specific problem columns
    your_worst_cols = ['327310', '221100', '327400', '562000', '481000']
    print("\n" + "="*70)
    print("ANALYSIS OF YOUR SPECIFIC WORST COLUMNS")
    print("="*70)
    
    for col in your_worst_cols:
        col_data = rel_df[rel_df['column'] == col]
        if len(col_data) > 0:
            row = col_data.iloc[0]
            print(f"\n{col}:")
            print(f"  MAPE: {row['MAPE_%']:.1f}%")
            print(f"  Mean true value: {row['mean_true']:.4f}")
            print(f"  MAE: {row['MAE']:.4f}")
            print(f"  Within 20%: {row['within_20%']:.1f}% of imputations")
            
            # Check if this is actually good relative to scale
            if row['mean_true'] > 1:
                expected_mae = row['mean_true'] * 0.1  # 10% error
                print(f"  10% error would be MAE={expected_mae:.4f}")
                print(f"  Actual MAE={row['MAE']:.4f} - {'✅ Good' if row['MAE'] <= expected_mae else '⚠️ Needs review'}")
    
    return rel_df


# # ============================================================================
# # EXECUTE ANALYSIS
# # ============================================================================

# # Run the full analysis
# rel_df = run_relative_error_analysis(
#     original_df=original_df,
#     imputed_df=imputed_df, 
#     masked_df=masked_df,
#     output_dir='hotdeck_results'
# )

# # Quick view of the worst columns by relative error
# print("\n" + "="*70)
# print("TOP 10 WORST COLUMNS BY RELATIVE ERROR")
# print("="*70)
# print(rel_df[['column', 'MAPE_%', 'mean_true', 'MAE', 'within_20%']].head(10).to_string())

# # Quick view of your specific columns
# print("\n" + "="*70)
# print("YOUR SPECIFIC COLUMNS")
# print("="*70)
# your_cols_df = rel_df[rel_df['column'].isin(['327310', '221100', '327400', '562000', '481000'])]
# print(your_cols_df[['column', 'MAPE_%', 'mean_true', 'MAE', 'within_20%']].to_string())