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

#------------------------------------------------------------------------------------------------------------------------------
#extra statistics stuff (country per user)

def list_missing_industries_by_country(df, top_n=None):
    """
    List missing industries (zero values) for each country.
    
    Parameters:
    df (DataFrame): CEDA data with 'Country' column and industry columns
    top_n (int, optional): Show only top N countries with most missing industries.
                          If None, shows all countries with missing data.
    
    Returns:
    DataFrame: Summary of missing industries per country
    dict: Detailed dictionary with missing industries for each country
    """
    # Get industry columns
    industry_cols = [col for col in df.columns if col != 'Country']
    
    # Initialize results
    missing_summary = []
    missing_details = {}
    
    # Check each country
    for idx, row in df.iterrows():
        country = row['Country']
        missing_industries = []
        
        # Find industries with zero values
        for industry in industry_cols:
            if row[industry] == 0:
                missing_industries.append(industry)
        
        # If country has missing industries, add to results
        if missing_industries:
            missing_count = len(missing_industries)
            missing_summary.append({
                'Country': country,
                'Missing_Count': missing_count,
                'Missing_Percentage': (missing_count / len(industry_cols)) * 100
            })
            missing_details[country] = {
                'count': missing_count,
                'percentage': (missing_count / len(industry_cols)) * 100,
                'industries': missing_industries
            }
    
    if not missing_summary:
        print("No missing industries (zero values) found in any country!")
        return None, None
    
    # Convert summary to DataFrame
    missing_summary_df = pd.DataFrame(missing_summary)
    
    # Sort by missing count (descending)
    missing_summary_df = missing_summary_df.sort_values('Missing_Count', ascending=False)
    
    # Limit to top_n if specified
    if top_n is not None and top_n > 0:
        missing_summary_df = missing_summary_df.head(top_n)
        top_countries = missing_summary_df['Country'].tolist()
        # Filter details dictionary to only include top countries
        missing_details = {k: v for k, v in missing_details.items() if k in top_countries}
    
    return missing_summary_df, missing_details

def print_missing_industries_report(missing_summary_df, missing_details, show_details=True, max_industries_per_country=10):
    """
    Print a formatted report of missing industries by country.
    
    Parameters:
    missing_summary_df (DataFrame): Summary DataFrame from list_missing_industries_by_country
    missing_details (dict): Details dictionary from list_missing_industries_by_country
    show_details (bool): Whether to show detailed list of missing industries
    max_industries_per_country (int): Maximum number of industries to show per country
    """
    if missing_summary_df is None:
        return
    
    print("=" * 80)
    print("MISSING INDUSTRIES BY COUNTRY REPORT")
    print("=" * 80)
    
    # Print summary statistics
    total_countries = len(missing_summary_df)
    total_missing = missing_summary_df['Missing_Count'].sum()
    avg_missing = missing_summary_df['Missing_Count'].mean()
    max_missing = missing_summary_df['Missing_Count'].max()
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   • Countries with missing data: {total_countries}")
    print(f"   • Total missing industry entries: {total_missing}")
    print(f"   • Average missing per country: {avg_missing:.1f}")
    print(f"   • Maximum missing in a country: {max_missing}")
    print(f"   • Average missing percentage: {missing_summary_df['Missing_Percentage'].mean():.1f}%")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Country':<25} {'Missing Count':<15} {'Missing %':<12} {'Severity'}")
    print(f"{'-'*80}")
    
    for _, row in missing_summary_df.iterrows():
        country = row['Country']
        count = row['Missing_Count']
        percentage = row['Missing_Percentage']
        
        # Determine severity level
        if percentage < 10:
            severity = "Low"
        elif percentage < 30:
            severity = "Medium"
        elif percentage < 50:
            severity = "High"
        else:
            severity = "Critical"
        
        print(f"{country:<25} {count:<15} {percentage:.1f}%{'':<6} {severity}")
    
    # Print detailed missing industries if requested
    if show_details and missing_details:
        print(f"\n{'='*80}")
        print("DETAILED MISSING INDUSTRIES BY COUNTRY:")
        print(f"{'='*80}")
        
        for country, details in missing_details.items():
            print(f"\n📍 {country}:")
            print(f"   • Missing industries: {details['count']} ({details['percentage']:.1f}% of total)")
            
            industries = details['industries']
            if len(industries) <= max_industries_per_country:
                print(f"   • Industries: {', '.join(industries)}")
            else:
                print(f"   • Industries (first {max_industries_per_country} of {len(industries)}):")
                print(f"     {', '.join(industries[:max_industries_per_country])}")
                print(f"     ... and {len(industries) - max_industries_per_country} more")
    
    print(f"\n{'='*80}")

def export_missing_industries_to_csv(missing_summary_df, missing_details, filename_prefix="missing_industries"):
    """
    Export missing industries data to CSV files.
    
    Parameters:
    missing_summary_df (DataFrame): Summary DataFrame
    missing_details (dict): Details dictionary
    filename_prefix (str): Prefix for output filenames
    """
    if missing_summary_df is None:
        return
    
    # Export summary
    summary_filename = f"{filename_prefix}_summary.csv"
    missing_summary_df.to_csv(summary_filename, index=False)
    print(f"✓ Summary exported to: {summary_filename}")
    
    # Export detailed data
    detailed_data = []
    for country, details in missing_details.items():
        for industry in details['industries']:
            detailed_data.append({
                'Country': country,
                'Industry': industry,
                'Missing_Count': details['count'],
                'Missing_Percentage': details['percentage']
            })
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_filename = f"{filename_prefix}_detailed.csv"
        detailed_df.to_csv(detailed_filename, index=False)
        print(f"✓ Detailed data exported to: {detailed_filename}")
    
    # Export wide format (country x industry matrix)
    if missing_details:
        # Get all unique industries with missing data
        all_missing_industries = set()
        for details in missing_details.values():
            all_missing_industries.update(details['industries'])
        
        # Create matrix
        matrix_data = []
        for country, details in missing_details.items():
            row = {'Country': country}
            for industry in sorted(all_missing_industries):
                row[industry] = 1 if industry in details['industries'] else 0
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        matrix_filename = f"{filename_prefix}_matrix.csv"
        matrix_df.to_csv(matrix_filename, index=False)
        print(f"✓ Matrix format exported to: {matrix_filename}")


# summary_df, details_dict = list_missing_industries_by_country(ceda_data)

# if summary_df is not None:
#     # Print report
#     print_missing_industries_report(summary_df, details_dict, show_details=True)
    
#     # Export to CSV
#     export_missing_industries_to_csv(summary_df, details_dict)
    
#     # Get only top 10 countries with most missing industries
#     top_summary, top_details = list_missing_industries_by_country(ceda_data, top_n=10)
#     print("\n" + "="*80)
#     print("TOP 10 COUNTRIES WITH MOST MISSING INDUSTRIES:")
#     print_missing_industries_report(top_summary, top_details, show_details=True, max_industries_per_country=5)
    
#     # You can also use the summary DataFrame for further analysis
#     print("\n📈 Quick analysis of missing data distribution:")
#     print(f"   • Countries with >50% missing: {len(summary_df[summary_df['Missing_Percentage'] > 50])}")
#     print(f"   • Countries with >25% missing: {len(summary_df[summary_df['Missing_Percentage'] > 25])}")
#     print(f"   • Countries with >10% missing: {len(summary_df[summary_df['Missing_Percentage'] > 10])}")

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