import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def match_countries(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Match countries between indicators and emissions datasets
    Returns matched DataFrames and matching statistics
    """
    # Clean country names by stripping whitespace
    indicators_df['Country Name'] = indicators_df['Country Name'].str.strip()
    emissions_df['Country'] = emissions_df['Country'].str.strip()
    
    # Get unique country lists
    indicators_countries = set(indicators_df['Country Name'].dropna().unique())
    emissions_countries = set(emissions_df['Country'].dropna().unique())
    
    # Find matches and mismatches
    matched_countries = sorted(list(indicators_countries.intersection(emissions_countries)))
    indicators_only = sorted(list(indicators_countries - emissions_countries))
    emissions_only = sorted(list(emissions_countries - indicators_countries))
    
    # Calculate matching statistics
    total_possible = len(indicators_countries.union(emissions_countries))
    match_percentage = (len(matched_countries) / total_possible * 100) if total_possible > 0 else 0
    
    stats = {
        'matched_countries': matched_countries,
        'indicators_only': indicators_only,
        'emissions_only': emissions_only,
        'match_percentage': match_percentage,
        'n_matched': len(matched_countries),
        'n_indicators_only': len(indicators_only),
        'n_emissions_only': len(emissions_only)
    }
    
    # Filter DataFrames to only include matched countries
    matched_indicators = indicators_df[indicators_df['Country Name'].isin(matched_countries)].copy()
    matched_emissions = emissions_df[emissions_df['Country'].isin(matched_countries)].copy()
    
    # Ensure consistent ordering by country
    matched_indicators = matched_indicators.sort_values('Country Name')
    matched_emissions = matched_emissions.sort_values('Country')
    
    return matched_indicators, matched_emissions, stats

def save_missing_countries(indicators_df, emissions_df, filename="missing_countries.xlsx"):
    """Save missing countries from both datasets to Excel."""
    ind = set(indicators_df['Country Name'].str.strip().dropna())
    em = set(emissions_df['Country'].str.strip().dropna())
    
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(sorted(ind - em), columns=['Only in Indicators']).to_excel(
            writer, sheet_name='Only_Indicators', index=False)
        pd.DataFrame(sorted(em - ind), columns=['Only in Emissions']).to_excel(
            writer, sheet_name='Only_Emissions', index=False)
    
    return filename


def calculate_correlations(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlations between all indicator columns and all industry emissions columns
    """
    # Extract indicator columns (excluding country name)
    indicator_columns = [col for col in indicators_df.columns if col != 'Country Name']
    
    # Extract industry columns (excluding country column)
    industry_columns = [col for col in emissions_df.columns if col != 'Country']
    
    # Create a dictionary to store correlations
    correlation_results = []
    
    # Calculate correlations for each industry
    for industry in industry_columns:
        industry_correlations = []
        
        for indicator in indicator_columns:
            # Merge data for this specific indicator-industry pair
            merged = pd.merge(
                indicators_df[['Country Name', indicator]],
                emissions_df[['Country', industry]],
                left_on='Country Name',
                right_on='Country'
            )
            
            # Drop rows with missing values
            merged_clean = merged[[indicator, industry]].dropna()
            
            if len(merged_clean) >= 3:  # Need at least 3 data points for meaningful correlation
                # Calculate Pearson correlation
                correlation = merged_clean[indicator].corr(merged_clean[industry])
                
                # Store result
                industry_correlations.append({
                    'Industry': industry,
                    'Indicator': indicator,
                    'Correlation': correlation,
                    'N_Countries': len(merged_clean)
                })
        
        # Add all correlations for this industry
        correlation_results.extend(industry_correlations)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(correlation_results)
    
    # Calculate absolute correlation for ranking
    results_df['Abs_Correlation'] = results_df['Correlation'].abs()
    
    return results_df

def calculate_correlations_optimized(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized version for larger datasets (400 industries × 200 countries)
    """
    # Merge the two datasets once
    merged = pd.merge(
        indicators_df,
        emissions_df,
        left_on='Country Name',
        right_on='Country',
        how='inner'
    )
    
    # Get column lists
    indicator_cols = [col for col in indicators_df.columns if col != 'Country Name']
    industry_cols = [col for col in emissions_df.columns if col != 'Country']
    
    # Pre-calculate correlation matrix
    correlation_matrix = pd.DataFrame(index=indicator_cols, columns=industry_cols)
    country_counts = pd.DataFrame(index=indicator_cols, columns=industry_cols)
    
    # Calculate correlations efficiently
    for indicator in indicator_cols:
        for industry in industry_cols:
            # Get valid pairs (non-NaN)
            valid_mask = merged[[indicator, industry]].notna().all(axis=1)
            valid_data = merged.loc[valid_mask, [indicator, industry]]
            
            if len(valid_data) >= 3:
                corr = valid_data[indicator].corr(valid_data[industry])
                correlation_matrix.loc[indicator, industry] = corr
                country_counts.loc[indicator, industry] = len(valid_data)
    
    # Convert to long format
    correlation_df = correlation_matrix.stack().reset_index()
    correlation_df.columns = ['Indicator', 'Industry', 'Correlation']
    
    # Add country counts
    counts_series = country_counts.stack()
    correlation_df['N_Countries'] = correlation_df.apply(
        lambda row: counts_series.loc[(row['Indicator'], row['Industry'])], axis=1
    )
    
    # Convert to numeric
    correlation_df['Correlation'] = pd.to_numeric(correlation_df['Correlation'])
    correlation_df['N_Countries'] = pd.to_numeric(correlation_df['N_Countries'])
    
    # Add absolute correlation
    correlation_df['Abs_Correlation'] = correlation_df['Correlation'].abs()
    
    return correlation_df

# Use this function instead of calculate_correlations for large datasets

def rank_indicators_by_correlation(correlation_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Rank indicators by their average absolute correlation across all industries
    """
    # Group by indicator and calculate average absolute correlation
    indicator_rankings = correlation_df.groupby('Indicator').agg({
        'Abs_Correlation': 'mean',
        'Correlation': lambda x: x.mean(),  # Average correlation (can be positive or negative)
        'Industry': 'count'  # Number of industries this indicator was correlated with
    }).rename(columns={'Industry': 'N_Industries'})
    
    # Sort by average absolute correlation (descending)
    indicator_rankings = indicator_rankings.sort_values('Abs_Correlation', ascending=False)
    
    return indicator_rankings.head(top_n)

def analyze_industry_specific_correlations(correlation_df: pd.DataFrame, top_per_industry: int = 5) -> Dict:
    """
    Get top correlations for each industry
    """
    industry_results = {}
    
    for industry in correlation_df['Industry'].unique():
        industry_data = correlation_df[correlation_df['Industry'] == industry]
        
        # Get top positive correlations
        top_positive = industry_data.nlargest(top_per_industry, 'Correlation')[['Indicator', 'Correlation', 'N_Countries']]
        
        # Get top negative correlations
        top_negative = industry_data.nsmallest(top_per_industry, 'Correlation')[['Indicator', 'Correlation', 'N_Countries']]
        
        industry_results[industry] = {
            'top_positive': top_positive,
            'top_negative': top_negative,
            'n_indicators': len(industry_data)
        }
    
    return industry_results

def main():
    # Load datasets
    print("Loading datasets...")
    
    # Load World Development Indicators
    indicators_df = pd.read_csv('country_indicator_latest_values.csv')
    
    # Load emissions data (CEDE cleaned)
    emissions_df = pd.read_excel('cedacleanedsmall.xlsx')
    
    print(f"Indicators dataset shape: {indicators_df.shape}")
    print(f"Emissions dataset shape: {emissions_df.shape}")
    print()
    
    # Step 1: Match countries
    print("Matching countries between datasets...")
    matched_indicators, matched_emissions, match_stats = match_countries(indicators_df, emissions_df)
    
    print(f"Match percentage: {match_stats['match_percentage']:.2f}%")
    print(f"Matched countries: {match_stats['n_matched']}")
    print(f"Countries only in indicators: {match_stats['n_indicators_only']}")
    print(f"Countries only in emissions: {match_stats['n_emissions_only']}")
    print()
    
    if match_stats['n_matched'] == 0:
        print("No countries matched! Cannot calculate correlations.")
        return
    
    # Display some unmatched examples
    if match_stats['indicators_only']:
        print("Sample countries only in indicators dataset:")
        for country in match_stats['indicators_only'][:10]:
            print(f"  - {country}")
        if len(match_stats['indicators_only']) > 10:
            print(f"  ... and {len(match_stats['indicators_only']) - 10} more")
    
    if match_stats['emissions_only']:
        print("\nSample countries only in emissions dataset:")
        for country in match_stats['emissions_only'][:10]:
            print(f"  - {country}")
        if len(match_stats['emissions_only']) > 10:
            print(f"  ... and {len(match_stats['emissions_only']) - 10} more")
    print()
    
    print(f"After matching: {len(matched_indicators)} countries in indicators")
    print(f"After matching: {len(matched_emissions)} countries in emissions")
    print()
    
    # Step 2: Calculate correlations
    print("Calculating correlations between indicators and industry emissions...")
    correlation_df = calculate_correlations(matched_indicators, matched_emissions)
    
    print(f"Calculated {len(correlation_df)} indicator-industry correlations")
    print()
    
    # Step 3: Rank indicators by average correlation across industries
    print("TOP 20 INDICATORS BY AVERAGE ABSOLUTE CORRELATION:")
    print("=" * 80)
    indicator_rankings = rank_indicators_by_correlation(correlation_df, top_n=20)
    
    for idx, (indicator, row) in enumerate(indicator_rankings.iterrows(), 1):
        avg_corr = row['Correlation']
        avg_abs_corr = row['Abs_Correlation']
        n_industries = row['N_Industries']
        direction = "positive" if avg_corr > 0 else "negative"
        
        print(f"{idx:2d}. {indicator:50s} | Avg correlation: {avg_corr:+.3f} ({direction})")
        print(f"    {'':50s}   Avg absolute: {avg_abs_corr:.3f} | Industries: {n_industries}")
    print()
    
    # Step 4: Show industry-specific top correlations
    print("INDUSTRY-SPECIFIC TOP CORRELATIONS:")
    print("=" * 80)
    industry_analysis = analyze_industry_specific_correlations(correlation_df, top_per_industry=3)
    
    for industry, results in industry_analysis.items():
        print(f"\nIndustry: {industry}")
        print(f"Total indicators analyzed: {results['n_indicators']}")
        
        print("  Top positive correlations:")
        for _, row in results['top_positive'].iterrows():
            print(f"    - {row['Indicator']}: {row['Correlation']:.3f} (n={row['N_Countries']})")
        
        print("  Top negative correlations:")
        for _, row in results['top_negative'].iterrows():
            print(f"    - {row['Indicator']}: {row['Correlation']:.3f} (n={row['N_Countries']})")
    
    # Optional: Save results to files
    print("\nSaving results to files...")
    correlation_df.to_csv('all_correlations.csv', index=False)
    indicator_rankings.to_csv('indicator_rankings.csv')
    
    # Save industry-specific results
    with pd.ExcelWriter('industry_specific_correlations.xlsx') as writer:
        for industry, results in industry_analysis.items():
            # Combine positive and negative
            combined = pd.concat([
                results['top_positive'].assign(Type='Positive'),
                results['top_negative'].assign(Type='Negative')
            ])
            combined.to_excel(writer, sheet_name=industry[:31], index=False)
    
    print("Analysis complete! Files saved:")
    print("  - all_correlations.csv: All individual correlations")
    print("  - indicator_rankings.csv: Top indicators by average correlation")
    print("  - industry_specific_correlations.xlsx: Top correlations per industry")

if __name__ == "__main__":
    main()