import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_zeros_per_country(df):
    """
    Create bar plot showing number of zeros per country.
    Shows all countries that have at least one zero value.
    
    Parameters:
    df (DataFrame): CEDA data with 'Country' column and industry columns
    """
    # Calculate zeros per country
    numeric_df = df.drop('Country', axis=1)
    zeros_per_country = {}
    
    for idx, row in df.iterrows():
        country = row['Country']
        zero_count = (row[numeric_df.columns] == 0).sum()
        if zero_count > 0:  # Only include countries with zeros
            zeros_per_country[country] = zero_count
    
    if not zeros_per_country:
        print("No zero values found in any country!")
        return None
    
    # Convert to DataFrame for plotting
    zeros_df = pd.DataFrame({
        'Country': list(zeros_per_country.keys()),
        'Zero_Count': list(zeros_per_country.values())
    })
    
    # Sort by zero count (descending)
    zeros_df = zeros_df.sort_values('Zero_Count', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(zeros_df) * 0.4), 8))
    
    # Create bar plot
    bars = ax.bar(range(len(zeros_df)), zeros_df['Zero_Count'], color='red', alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Number of Zero Values', fontsize=12)
    ax.set_title(f'Zero Values per Country ({len(zeros_df)} countries with zeros)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks
    ax.set_xticks(range(len(zeros_df)))
    ax.set_xticklabels(zeros_df['Country'], rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, zeros_df['Zero_Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add horizontal line for average
    avg_zeros = zeros_df['Zero_Count'].mean()
    ax.axhline(y=avg_zeros, color='blue', linestyle='--', alpha=0.7, 
               label=f'Average: {avg_zeros:.1f} zeros')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return zeros_df

def plot_zeros_per_industry(df):
    """
    Create bar plot showing number of zeros per industry.
    Shows all industries that have at least one zero value.
    
    Parameters:
    df (DataFrame): CEDA data with 'Country' column and industry columns
    """
    # Calculate zeros per industry
    numeric_df = df.drop('Country', axis=1)
    zeros_per_industry = {}
    
    for col in numeric_df.columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:  # Only include industries with zeros
            zeros_per_industry[col] = zero_count
    
    if not zeros_per_industry:
        print("No zero values found in any industry!")
        return None
    
    # Convert to DataFrame for plotting
    zeros_df = pd.DataFrame({
        'Industry': list(zeros_per_industry.keys()),
        'Zero_Count': list(zeros_per_industry.values())
    })
    
    # Sort by zero count (descending)
    zeros_df = zeros_df.sort_values('Zero_Count', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, len(zeros_df) * 0.3), 8))
    
    # Create bar plot with color gradient
    bars = ax.bar(range(len(zeros_df)), zeros_df['Zero_Count'])
    
    # Customize plot
    ax.set_xlabel('Industry/Sector Code', fontsize=12)
    ax.set_ylabel('Number of Zero Values', fontsize=12)
    ax.set_title(f'Zero Values per Industry ({len(zeros_df)} industries with zeros)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks
    ax.set_xticks(range(len(zeros_df)))
    ax.set_xticklabels(zeros_df['Industry'], rotation=90, ha='center', fontsize=9)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, zeros_df['Zero_Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(value)}', ha='center', va='bottom', fontsize=8)
    
    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add horizontal line for average
    avg_zeros = zeros_df['Zero_Count'].mean()
    ax.axhline(y=avg_zeros, color='blue', linestyle='--', alpha=0.7, 
               label=f'Average: {avg_zeros:.1f} zeros')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return zeros_df

# Example usage:
if __name__ == "__main__":
    # Assuming ceda_data is your DataFrame
    # Plot zeros per country
    country_zeros = plot_zeros_per_country(ceda_data)
    
    if country_zeros is not None:
        print(f"\nFound {len(country_zeros)} countries with zero values")
        print(f"Total zero values across all countries: {country_zeros['Zero_Count'].sum()}")
        print(f"\nTop 10 countries with most zeros:")
        print(country_zeros.head(10))
    
    # Plot zeros per industry
    industry_zeros = plot_zeros_per_industry(ceda_data)
    
    if industry_zeros is not None:
        print(f"\nFound {len(industry_zeros)} industries with zero values")
        print(f"Total zero values across all industries: {industry_zeros['Zero_Count'].sum()}")
        print(f"\nTop 10 industries with most zeros:")
        print(industry_zeros.head(10))