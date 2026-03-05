import pandas as pd
import numpy as np

# Load the datasets
# 1. CEDA cleaned data (country x industry matrix)
ceda_df = pd.read_csv('ceda_cleaned.csv')

# 2. Carbon intensity indicators
carbon_df = pd.read_excel('xx_country_variables/carbon_intensity_indicators_values.xlsx', sheet_name='Indicator Values')

# 3. Industry ratings
industry_df = pd.read_csv('xx_industry_variables/industry_ratings_output.csv')

# 4. Geo CEPII data
geo_df = pd.read_excel('data/geo_cepii.xls', sheet_name='geo_cepii')


# ============================================================================
# DATA PREPARATION
# ============================================================================

# 1. Reshape CEDA data from wide to long format
# Identify industry columns (all columns after Country Code)
industry_cols = ceda_df.columns[2:].tolist()

# Melt the dataframe
ceda_long = pd.melt(
    ceda_df,
    id_vars=['Country', 'Country Code'],
    value_vars=industry_cols,
    var_name='industry_code',
    value_name='carbon_intensity'
)

# Remove rows with missing carbon intensity values
ceda_long = ceda_long.dropna(subset=['carbon_intensity'])

print(f"CEDA long format: {len(ceda_long)} rows")


# 2. Prepare carbon intensity indicators data
# First, check the actual column names in your carbon dataframe
print("\nCarbon dataframe columns:")
print(carbon_df.columns.tolist())

# Select relevant columns - using the exact column names from your data
carbon_cols = [
    'Country Code', 
    'GDP per capita, PPP (constant 2021 international $)',
    'Industry (including construction), value added (% of GDP)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Electricity production from coal sources (% of total)',
    'Energy intensity level of primary energy (MJ/$2021 PPP GDP)',
    'GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)',
    'Urban population (% of total population)',
    'Total natural resources rents (% of GDP)'
]

# Rename columns for easier access
carbon_renamed = {
    'GDP per capita, PPP (constant 2021 international $)': 'gdp_per_capita_ppp',
    'Industry (including construction), value added (% of GDP)': 'industry_value_added_pct',
    'Renewable energy consumption (% of total final energy consumption)': 'renewable_energy_pct',
    'Electricity production from coal sources (% of total)': 'coal_electricity_pct',
    'Energy intensity level of primary energy (MJ/$2021 PPP GDP)': 'energy_intensity_level',  # Changed to match actual column name
    'GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)': 'gdp_per_energy_unit',
    'Urban population (% of total population)': 'urban_population_pct',
    'Total natural resources rents (% of GDP)': 'natural_resources_rents_pct'
}

carbon_df_clean = carbon_df[carbon_cols].copy()
carbon_df_clean = carbon_df_clean.rename(columns=carbon_renamed)

print(f"\nCarbon indicators: {len(carbon_df_clean)} countries")
print("Cleaned carbon columns:", carbon_df_clean.columns.tolist())


# 3. Prepare industry ratings data
print("\nIndustry ratings columns:")
print(industry_df.columns.tolist())

# Select relevant columns
industry_cols_selected = [
    'industry_code', 'industry_name',
    'process_emission_intensity_score', 'material_processing_depth_score',
    'thermal_process_intensity_score', 'electrification_feasibility_score',
    'continuous_operations_intensity_score', 'material_throughput_scale_score',
    'chemical_intensity_score', 'capital_vs_labor_intensity_score'
]

industry_df_clean = industry_df[industry_cols_selected].copy()

print(f"Industry ratings: {len(industry_df_clean)} industries")


# 4. Prepare geo data
print("\nGeo data columns:")
print(geo_df.columns.tolist())

# Select relevant columns and rename
geo_cols = ['iso3', 'area', 'dis_int', 'lat', 'lon']
geo_renamed = {
    'iso3': 'Country Code',
    'area': 'area_sq_km',
    'dis_int': 'distance_international_km'
}

geo_df_clean = geo_df[geo_cols].copy()
geo_df_clean = geo_df_clean.rename(columns=geo_renamed)

# Remove duplicates (multiple cities per country)
geo_df_clean = geo_df_clean.drop_duplicates(subset=['Country Code'])

print(f"Geo data: {len(geo_df_clean)} countries")


# ============================================================================
# MERGE DATASETS
# ============================================================================

# Start with CEDA long format
merged_df = ceda_long.copy()

# Merge with carbon indicators
merged_df = merged_df.merge(
    carbon_df_clean,
    on='Country Code',
    how='inner',  # Only keep matches
    indicator='match_carbon'
)

# Merge with geo data
merged_df = merged_df.merge(
    geo_df_clean,
    on='Country Code',
    how='inner',  # Only keep matches
    indicator='match_geo'
)

# Merge with industry ratings
merged_df = merged_df.merge(
    industry_df_clean,
    on='industry_code',
    how='inner',  # Only keep matches
    indicator='match_industry'
)

print(f"\nAfter merges: {len(merged_df)} rows")
print("Merged dataframe columns:", merged_df.columns.tolist())


# ============================================================================
# CREATE MATCHING STATISTICS
# ============================================================================

# Create a summary of matching statistics
matching_stats = {}

# 1. Country matching
countries_in_ceda = set(ceda_df['Country Code'].unique())
countries_in_carbon = set(carbon_df_clean['Country Code'].unique())
countries_in_geo = set(geo_df_clean['Country Code'].unique())
industries_in_ceda = set(ceda_long['industry_code'].unique())
industries_in_rating = set(industry_df_clean['industry_code'].unique())

matching_stats['total_countries_in_ceda'] = len(countries_in_ceda)
matching_stats['total_countries_in_carbon'] = len(countries_in_carbon)
matching_stats['total_countries_in_geo'] = len(countries_in_geo)
matching_stats['countries_matched_all'] = len(set(countries_in_ceda) & set(countries_in_carbon) & set(countries_in_geo))

matching_stats['total_industries_in_ceda'] = len(industries_in_ceda)
matching_stats['total_industries_in_rating'] = len(industries_in_rating)
matching_stats['industries_matched'] = len(industries_in_ceda & industries_in_rating)

matching_stats['initial_ceda_rows'] = len(ceda_long)
matching_stats['after_carbon_match'] = len(merged_df[merged_df['match_carbon'] == 'both'])
matching_stats['after_geo_match'] = len(merged_df[merged_df['match_geo'] == 'both'])
matching_stats['after_industry_match'] = len(merged_df[merged_df['match_industry'] == 'both'])
matching_stats['final_rows'] = len(merged_df)

# Country-level matching details
country_match_details = []
for country in sorted(countries_in_ceda):
    in_carbon = country in countries_in_carbon
    in_geo = country in countries_in_geo
    rows_in_ceda = len(ceda_long[ceda_long['Country Code'] == country])
    rows_in_final = len(merged_df[merged_df['Country Code'] == country]) if country in merged_df['Country Code'].values else 0
    
    country_match_details.append({
        'Country Code': country,
        'Country Name': ceda_df[ceda_df['Country Code'] == country]['Country'].iloc[0] if country in ceda_df['Country Code'].values else 'Unknown',
        'In Carbon Indicators': in_carbon,
        'In Geo Data': in_geo,
        'Rows in CEDA': rows_in_ceda,
        'Rows in Final Dataset': rows_in_final
    })

country_match_df = pd.DataFrame(country_match_details)

# Industry-level matching details
industry_match_details = []
for industry in sorted(industries_in_ceda):
    in_rating = industry in industries_in_rating
    rows_in_ceda = len(ceda_long[ceda_long['industry_code'] == industry])
    rows_in_final = len(merged_df[merged_df['industry_code'] == industry]) if industry in merged_df['industry_code'].values else 0
    
    # Get industry name if available
    industry_name = ''
    if in_rating:
        industry_name = industry_df_clean[industry_df_clean['industry_code'] == industry]['industry_name'].iloc[0]
    
    industry_match_details.append({
        'Industry Code': industry,
        'Industry Name': industry_name,
        'In Industry Ratings': in_rating,
        'Rows in CEDA': rows_in_ceda,
        'Rows in Final Dataset': rows_in_final
    })

industry_match_df = pd.DataFrame(industry_match_details)


# ============================================================================
# CREATE FINAL DATASET
# ============================================================================

# Select final columns - use the correct column names
final_columns = [
    'Country', 'Country Code', 'industry_code', 'industry_name',
    'carbon_intensity',
    # WDI variables
    'gdp_per_capita_ppp', 'industry_value_added_pct', 'renewable_energy_pct',
    'coal_electricity_pct', 'energy_intensity_level',  # Changed to match actual column name
    'gdp_per_energy_unit', 'urban_population_pct', 'natural_resources_rents_pct',
    # Industry variables
    'process_emission_intensity_score', 'material_processing_depth_score',
    'thermal_process_intensity_score', 'electrification_feasibility_score',
    'continuous_operations_intensity_score', 'material_throughput_scale_score',
    'chemical_intensity_score', 'capital_vs_labor_intensity_score',
    # Geographic variables
    'area_sq_km', 'distance_international_km', 'lat', 'lon'
]

# Check which columns are actually in merged_df
available_columns = [col for col in final_columns if col in merged_df.columns]
missing_columns = [col for col in final_columns if col not in merged_df.columns]

print(f"\nAvailable columns in final dataset: {len(available_columns)}")
print(f"Missing columns: {missing_columns}")

final_dataset = merged_df[available_columns].copy()

# Drop the match indicator columns if they exist
for col in ['match_carbon', 'match_geo', 'match_industry']:
    if col in final_dataset.columns:
        final_dataset = final_dataset.drop(columns=[col])


# ============================================================================
# SAVE OUTPUT FILES
# ============================================================================

# Save final dataset
final_dataset.to_csv('combined_carbon_intensity_dataset.csv', index=False)
print(f"\nFinal dataset saved: {len(final_dataset)} rows, {len(final_dataset.columns)} columns")

# Save matching statistics summary
stats_df = pd.DataFrame([matching_stats]).T.reset_index()
stats_df.columns = ['Metric', 'Value']
stats_df.to_excel('matching_statistics_summary.xlsx', index=False)

# Save detailed country matching
country_match_df.to_excel('country_matching_details.xlsx', index=False)

# Save detailed industry matching
industry_match_df.to_excel('industry_matching_details.xlsx', index=False)

# Create a comprehensive matching report
with pd.ExcelWriter('matching_statistics_complete.xlsx') as writer:
    # Summary sheet
    pd.DataFrame([matching_stats]).T.reset_index().to_excel(
        writer, sheet_name='Summary', index=False, header=['Metric', 'Value']
    )
    
    # Country details sheet
    country_match_df.to_excel(writer, sheet_name='Country Details', index=False)
    
    # Industry details sheet
    industry_match_df.to_excel(writer, sheet_name='Industry Details', index=False)
    
    # Sample of final dataset
    final_dataset.head(100).to_excel(writer, sheet_name='Sample Data (first 100)', index=False)


# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("MATCHING STATISTICS SUMMARY")
print("="*60)
print(f"Total countries in CEDA: {matching_stats['total_countries_in_ceda']}")
print(f"Total countries in Carbon indicators: {matching_stats['total_countries_in_carbon']}")
print(f"Total countries in Geo data: {matching_stats['total_countries_in_geo']}")
print(f"Countries matched across all datasets: {matching_stats['countries_matched_all']}")
print()
print(f"Total industries in CEDA: {matching_stats['total_industries_in_ceda']}")
print(f"Total industries in Ratings: {matching_stats['total_industries_in_rating']}")
print(f"Industries matched: {matching_stats['industries_matched']}")
print()
print(f"Initial CEDA rows: {matching_stats['initial_ceda_rows']:,}")
print(f"After carbon match: {matching_stats['after_carbon_match']:,}")
print(f"After geo match: {matching_stats['after_geo_match']:,}")
print(f"After industry match: {matching_stats['after_industry_match']:,}")
print(f"Final dataset rows: {matching_stats['final_rows']:,}")
print("="*60)

# Show sample of countries that didn't match
unmatched_countries = []
for country in countries_in_ceda:
    if country not in countries_in_carbon or country not in countries_in_geo:
        unmatched_countries.append({
            'Country Code': country,
            'Country Name': ceda_df[ceda_df['Country Code'] == country]['Country'].iloc[0] if country in ceda_df['Country Code'].values else 'Unknown',
            'In Carbon': country in countries_in_carbon,
            'In Geo': country in countries_in_geo
        })

if unmatched_countries:
    print("\nCountries in CEDA not fully matched:")
    unmatched_df = pd.DataFrame(unmatched_countries[:20])  # Show first 20
    print(unmatched_df.to_string(index=False))
    if len(unmatched_countries) > 20:
        print(f"... and {len(unmatched_countries) - 20} more")

# Show sample of industries that didn't match
unmatched_industries = [ind for ind in industries_in_ceda if ind not in industries_in_rating]
if unmatched_industries:
    print(f"\nIndustries in CEDA not in ratings ({len(unmatched_industries)}):")
    print(unmatched_industries[:20], "..." if len(unmatched_industries) > 20 else "")