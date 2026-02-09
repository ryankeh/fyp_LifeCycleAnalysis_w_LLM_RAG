import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def match_countries(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame,
                   indicators_country_col: str = 'Country Code',
                   emissions_country_col: str = 'Country Code',
                   indicators_name_col: str = 'Country Name',
                   emissions_name_col: str = 'Country') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Match countries between indicators and emissions datasets
    Returns matched DataFrames and comprehensive matching statistics
    """
    # Ensure we have copies to avoid modifying originals
    indicators_df = indicators_df.copy()
    emissions_df = emissions_df.copy()
    
    # Clean country codes
    indicators_df[indicators_country_col] = indicators_df[indicators_country_col].astype(str).str.strip().str.upper()
    emissions_df[emissions_country_col] = emissions_df[emissions_country_col].astype(str).str.strip().str.upper()
    
    # Clean country names if provided
    if indicators_name_col in indicators_df.columns:
        indicators_df[indicators_name_col] = indicators_df[indicators_name_col].astype(str).str.strip()
    if emissions_name_col in emissions_df.columns:
        emissions_df[emissions_name_col] = emissions_df[emissions_name_col].astype(str).str.strip()
    
    # Get unique country lists
    indicators_codes = set(indicators_df[indicators_country_col].dropna().unique())
    emissions_codes = set(emissions_df[emissions_country_col].dropna().unique())
    
    # Find matches and mismatches
    matched_codes = sorted(list(indicators_codes.intersection(emissions_codes)))
    indicators_only_codes = sorted(list(indicators_codes - emissions_codes))
    emissions_only_codes = sorted(list(emissions_codes - indicators_codes))
    
    # Create lookup dictionaries for names
    indicators_name_lookup = dict(zip(
        indicators_df[indicators_country_col], 
        indicators_df[indicators_name_col] if indicators_name_col in indicators_df.columns 
        else indicators_df[indicators_country_col]
    ))
    
    emissions_name_lookup = dict(zip(
        emissions_df[emissions_country_col], 
        emissions_df[emissions_name_col] if emissions_name_col in emissions_df.columns 
        else emissions_df[emissions_country_col]
    ))
    
    # Create lists with both codes and names
    matched_countries = []
    for code in matched_codes:
        name = indicators_name_lookup.get(code, emissions_name_lookup.get(code, code))
        matched_countries.append({'code': code, 'name': name})
    
    indicators_only = []
    for code in indicators_only_codes:
        name = indicators_name_lookup.get(code, code)
        indicators_only.append({'code': code, 'name': name})
    
    emissions_only = []
    for code in emissions_only_codes:
        name = emissions_name_lookup.get(code, code)
        emissions_only.append({'code': code, 'name': name})
    
    # Calculate comprehensive statistics
    total_indicators = len(indicators_codes)
    total_emissions = len(emissions_codes)
    total_unique = len(indicators_codes.union(emissions_codes))
    
    # Match percentages (relative to each dataset)
    match_percentage_in_indicators = (len(matched_codes) / total_indicators * 100) if total_indicators > 0 else 0
    match_percentage_in_emissions = (len(matched_codes) / total_emissions * 100) if total_emissions > 0 else 0
    overall_match_percentage = (len(matched_codes) / total_unique * 100) if total_unique > 0 else 0
    
    stats = {
        # Basic counts
        'n_matched': len(matched_codes),
        'n_indicators_only': len(indicators_only_codes),
        'n_emissions_only': len(emissions_only_codes),
        'total_indicators': total_indicators,
        'total_emissions': total_emissions,
        'total_unique': total_unique,
        
        # Match percentages (focus on CEDA emissions dataset)
        'match_percentage_in_emissions': match_percentage_in_emissions,  # Primary metric
        'match_percentage_in_indicators': match_percentage_in_indicators,
        'overall_match_percentage': overall_match_percentage,
        
        # Country lists with names
        'matched_countries': matched_countries,
        'indicators_only': indicators_only,
        'emissions_only': emissions_only,
        
        # Raw codes (for backward compatibility)
        'matched_codes': matched_codes,
        'indicators_only_codes': indicators_only_codes,
        'emissions_only_codes': emissions_only_codes,
        
        # Column names used
        'indicators_country_col': indicators_country_col,
        'emissions_country_col': emissions_country_col,
        'indicators_name_col': indicators_name_col,
        'emissions_name_col': emissions_name_col
    }
    
    # Filter DataFrames to only include matched countries
    matched_indicators = indicators_df[indicators_df[indicators_country_col].isin(matched_codes)].copy()
    matched_emissions = emissions_df[emissions_df[emissions_country_col].isin(matched_codes)].copy()
    
    # Sort
    matched_indicators = matched_indicators.sort_values(indicators_country_col)
    matched_emissions = matched_emissions.sort_values(emissions_country_col)
    
    return matched_indicators, matched_emissions, stats

def save_missing_countries(indicators_df, emissions_df, 
                          indicators_country_col: str = 'Country Code',
                          emissions_country_col: str = 'Country Code',
                          indicators_name_col: str = 'Country Name',
                          emissions_name_col: str = 'Country',
                          filename="missing_countries.xlsx"):
    """
    Save missing countries from both datasets to Excel with both codes and names.
    """
    # Ensure we have copies
    indicators_df = indicators_df.copy()
    emissions_df = emissions_df.copy()
    
    # Clean data
    indicators_df[indicators_country_col] = indicators_df[indicators_country_col].astype(str).str.strip().str.upper()
    emissions_df[emissions_country_col] = emissions_df[emissions_country_col].astype(str).str.strip().str.upper()
    
    if indicators_name_col in indicators_df.columns:
        indicators_df[indicators_name_col] = indicators_df[indicators_name_col].astype(str).str.strip()
    if emissions_name_col in emissions_df.columns:
        emissions_df[emissions_name_col] = emissions_df[emissions_name_col].astype(str).str.strip()
    
    # Get unique sets
    ind_codes = set(indicators_df[indicators_country_col].dropna().unique())
    em_codes = set(emissions_df[emissions_country_col].dropna().unique())
    
    # Find missing codes
    indicators_only_codes = sorted(list(ind_codes - em_codes))
    emissions_only_codes = sorted(list(em_codes - ind_codes))
    
    # Create lookup dictionaries for names
    ind_name_lookup = dict(zip(
        indicators_df[indicators_country_col], 
        indicators_df[indicators_name_col] if indicators_name_col in indicators_df.columns 
        else indicators_df[indicators_country_col]
    ))
    
    em_name_lookup = dict(zip(
        emissions_df[emissions_country_col], 
        emissions_df[emissions_name_col] if emissions_name_col in emissions_df.columns 
        else emissions_df[emissions_country_col]
    ))
    
    # Prepare DataFrames with both code and name
    indicators_only_data = []
    for code in indicators_only_codes:
        name = ind_name_lookup.get(code, 'N/A')
        indicators_only_data.append({
            'Country_Code': code,
            'Country_Name': name
        })
    
    emissions_only_data = []
    for code in emissions_only_codes:
        name = em_name_lookup.get(code, 'N/A')
        emissions_only_data.append({
            'Country_Code': code,
            'Country_Name': name
        })
    
    # Create DataFrames
    indicators_only_df = pd.DataFrame(indicators_only_data)
    emissions_only_df = pd.DataFrame(emissions_only_data)
    
    # Also create a summary sheet
    summary_data = {
        'Metric': [
            'Total countries in indicators dataset',
            'Total countries in emissions (CEDA) dataset',
            'Countries matched between datasets',
            'Countries only in indicators dataset',
            'Countries only in emissions (CEDA) dataset',
            'Match percentage (relative to CEDA dataset)',
            'Coverage of CEDA dataset by indicators'
        ],
        'Count': [
            len(ind_codes),
            len(em_codes),
            len(ind_codes.intersection(em_codes)),
            len(indicators_only_codes),
            len(emissions_only_codes),
            f"{len(ind_codes.intersection(em_codes)) / len(em_codes) * 100:.1f}%",
            f"{len(ind_codes.intersection(em_codes)) / len(ind_codes) * 100:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        indicators_only_df.to_excel(writer, sheet_name='Only_In_Indicators', index=False)
        emissions_only_df.to_excel(writer, sheet_name='Only_In_Emissions', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✓ Country comparison exported to: {filename}")
    print(f"   • Summary sheet: Overall statistics")
    print(f"   • Only_In_Indicators: {len(indicators_only_df)} countries only in indicators")
    print(f"   • Only_In_Emissions: {len(emissions_only_df)} countries only in emissions")
    
    return filename, indicators_only_df, emissions_only_df

def print_matching_report(match_stats: Dict, focus_on_ceda: bool = True):
    """
    Print a comprehensive matching report, focused on CEDA dataset if requested.
    """
    print("=" * 80)
    print("COUNTRY MATCHING REPORT")
    print("=" * 80)
    
    # Focus on CEDA dataset statistics
    if focus_on_ceda:
        print(f"\n📊 FOCUS ON CEDA DATASET:")
        print(f"   • Countries in CEDA dataset: {match_stats['total_emissions']}")
        print(f"   • CEDA countries matched with indicators: {match_stats['n_matched']}")
        print(f"   • CEDA coverage by indicators: {match_stats['match_percentage_in_emissions']:.1f}%")
        print(f"   • CEDA countries NOT in indicators: {match_stats['n_emissions_only']}")
        
        if match_stats['n_emissions_only'] > 0:
            print(f"\n📍 CEDA Countries missing from indicators:")
            for i, country in enumerate(match_stats['emissions_only'][:10], 1):
                print(f"   {i:2d}. {country['code']} - {country['name']}")
            if len(match_stats['emissions_only']) > 10:
                print(f"   ... and {len(match_stats['emissions_only']) - 10} more")
    else:
        # Original balanced view
        print(f"\n📊 OVERALL MATCHING STATISTICS:")
        print(f"   • Total unique countries: {match_stats['total_unique']}")
        print(f"   • Countries matched: {match_stats['n_matched']} ({match_stats['overall_match_percentage']:.1f}% overlap)")
        print(f"   • Countries only in indicators: {match_stats['n_indicators_only']}")
        print(f"   • Countries only in CEDA: {match_stats['n_emissions_only']}")
        print(f"   • Match relative to CEDA: {match_stats['match_percentage_in_emissions']:.1f}%")
        print(f"   • Match relative to indicators: {match_stats['match_percentage_in_indicators']:.1f}%")
    
    print(f"\n📈 MATCHING EFFICIENCY:")
    print(f"   • CEDA dataset coverage: {match_stats['match_percentage_in_emissions']:.1f}%")
    if match_stats['match_percentage_in_emissions'] < 50:
        print(f"   ⚠️  Warning: Low coverage of CEDA dataset (<50%)")
    elif match_stats['match_percentage_in_emissions'] < 80:
        print(f"   ⚠️  Note: Moderate coverage of CEDA dataset (<80%)")
    else:
        print(f"   ✅ Good: High coverage of CEDA dataset (≥80%)")
    
    # Quick analysis
    print(f"\n🔍 ANALYSIS:")
    if match_stats['n_emissions_only'] > match_stats['n_indicators_only']:
        print(f"   • CEDA dataset has more unique countries (+{match_stats['n_emissions_only'] - match_stats['n_indicators_only']})")
    elif match_stats['n_indicators_only'] > match_stats['n_emissions_only']:
        print(f"   • Indicators dataset has more unique countries (+{match_stats['n_indicators_only'] - match_stats['n_emissions_only']})")
    else:
        print(f"   • Both datasets have the same number of unique unmached countries")
    
    print("\n" + "=" * 80)

#----------------------------------------------------------------------------------------------------------------------------












def calculate_correlations(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame,
                          indicators_country_col: str = 'Country Code',
                          emissions_country_col: str = 'Country Code') -> pd.DataFrame:
    """
    Calculate correlations between all indicator columns and all industry emissions columns
    """
    # Extract indicator columns (excluding country code)
    indicator_columns = [col for col in indicators_df.columns if col != indicators_country_col]
    
    # Extract industry columns (excluding country column)
    industry_columns = [col for col in emissions_df.columns if col != emissions_country_col]
    
    # Create a dictionary to store correlations
    correlation_results = []
    
    # Calculate correlations for each industry
    for industry in industry_columns:
        industry_correlations = []
        
        for indicator in indicator_columns:
            # Merge data for this specific indicator-industry pair
            merged = pd.merge(
                indicators_df[[indicators_country_col, indicator]],
                emissions_df[[emissions_country_col, industry]],
                left_on=indicators_country_col,
                right_on=emissions_country_col
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

def calculate_correlations_optimized(indicators_df: pd.DataFrame, emissions_df: pd.DataFrame,
                          indicators_country_col: str = 'Country Code',
                          emissions_country_col: str = 'Country Code',
                          indicators_name_col: str = 'Country Name',  # ADD THIS PARAMETER
                          emissions_name_col: str = 'Country') -> pd.DataFrame:  # ADD THIS PARAMETER
    """
    Calculate correlations between all indicator columns and all industry emissions columns
    """
    # Extract indicator columns (excluding ALL country-related columns)
    indicator_cols_to_exclude = [indicators_country_col]
    if indicators_name_col in indicators_df.columns:
        indicator_cols_to_exclude.append(indicators_name_col)
    
    indicator_columns = [col for col in indicators_df.columns if col not in indicator_cols_to_exclude]
    
    # Extract industry columns (excluding ALL country-related columns)
    industry_cols_to_exclude = [emissions_country_col]
    if emissions_name_col in emissions_df.columns:
        industry_cols_to_exclude.append(emissions_name_col)
    
    industry_columns = [col for col in emissions_df.columns if col not in industry_cols_to_exclude]
    
    # Create a dictionary to store correlations
    correlation_results = []
    
    # Calculate correlations for each industry
    for industry in industry_columns:
        industry_correlations = []
        
        for indicator in indicator_columns:
            # Merge data for this specific indicator-industry pair
            merged = pd.merge(
                indicators_df[[indicators_country_col, indicator]],
                emissions_df[[emissions_country_col, industry]],
                left_on=indicators_country_col,
                right_on=emissions_country_col
            )
            
            # Drop rows with missing values
            merged_clean = merged[[indicator, industry]].dropna()
            
            if len(merged_clean) >= 3:  # Need at least 3 data points for meaningful correlation
                try:
                    # Calculate Pearson correlation
                    correlation = merged_clean[indicator].corr(merged_clean[industry])
                    
                    # Store result
                    industry_correlations.append({
                        'Industry': industry,
                        'Indicator': indicator,
                        'Correlation': correlation,
                        'N_Countries': len(merged_clean)
                    })
                except (ValueError, TypeError):
                    # Skip if correlation calculation fails
                    continue
        
        # Add all correlations for this industry
        correlation_results.extend(industry_correlations)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(correlation_results)
    
    # Calculate absolute correlation for ranking
    results_df['Abs_Correlation'] = results_df['Correlation'].abs()
    
    return results_df

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