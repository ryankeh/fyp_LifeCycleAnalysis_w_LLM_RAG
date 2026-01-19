import numpy as np
import pandas as pd

def lookup_value(df, country, sector_code):
    """
    Look up a specific value for a country and sector code.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    country (str): Country name (exact match)
    sector_code (str): Sector code (exact column name)
    
    Returns:
    float or None: The value if found, None if not found
    """
    try:
        # Find the row for the country
        country_row = df[df['Country'] == country]
        
        if country_row.empty:
            print(f"Country '{country}' not found in data")
            return None
        
        # Check if sector_code exists as a column
        if sector_code not in df.columns:
            print(f"Sector code '{sector_code}' not found in columns")
            return None
        
        # Get the value
        value = country_row[sector_code].iloc[0]
        return value
        
    except Exception as e:
        print(f"Error looking up value: {e}")
        return None

def lookup_sector_values(df, sector_code):
    """
    Get all values for a specific sector across all countries.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    sector_code (str): Sector code (exact column name)
    
    Returns:
    Series or None: Series of values with country as index, None if error
    """
    try:
        if sector_code not in df.columns:
            print(f"Sector code '{sector_code}' not found in columns")
            return None
        
        return df.set_index('Country')[sector_code]
        
    except Exception as e:
        print(f"Error looking up sector values: {e}")
        return None

def lookup_country_values(df, country):
    """
    Get all sector values for a specific country.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    country (str): Country name (exact match)
    
    Returns:
    Series or None: Series of values with sector codes as index, None if error
    """
    try:
        # Find the row for the country
        country_row = df[df['Country'] == country]
        
        if country_row.empty:
            print(f"Country '{country}' not found in data")
            return None
        
        # Transpose to get sectors as index
        values = country_row.set_index('Country').T.iloc[:, 0]
        return values
        
    except Exception as e:
        print(f"Error looking up country values: {e}")
        return None

def find_missing_values(df):
    """
    Identify all missing values (NaN) in the DataFrame.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    
    Returns:
    dict: Dictionary with statistics and missing value locations
    """
    # Count missing values per column
    missing_per_column = df.isnull().sum()
    
    # Count missing values per row
    missing_per_country = df.set_index('Country').isnull().sum(axis=1)
    
    # Total missing values
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells) * 100
    
    # Find exact locations of missing values
    missing_locations = []
    for col in df.columns:
        if col == 'Country':
            continue
        missing_indices = df[df[col].isnull()].index
        for idx in missing_indices:
            country = df.loc[idx, 'Country']
            missing_locations.append({
                'country': country,
                'sector_code': col,
                'row_index': idx
            })
    
    return {
        'total_missing': total_missing,
        'total_cells': total_cells,
        'missing_percentage': missing_percentage,
        'missing_per_column': missing_per_column[missing_per_column > 0],
        'missing_per_country': missing_per_country[missing_per_country > 0],
        'missing_locations': missing_locations
    }

# Advanced function for batch lookups
def batch_lookup(df, country_sector_pairs):
    """
    Look up multiple country-sector pairs at once.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    country_sector_pairs (list): List of tuples [(country, sector_code), ...]
    
    Returns:
    dict: Dictionary with results for each lookup
    """
    results = {}
    
    for country, sector_code in country_sector_pairs:
        value = lookup_value(df, country, sector_code)
        results[(country, sector_code)] = {
            'value': value,
            'found': value is not None and not pd.isna(value)
        }
    
    return results

#----------------------------------------------------------------------------------------------

def analyze_zeros_and_missing(df):
    """
    Comprehensive analysis of zeros, missing values, and their patterns.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    
    Returns:
    dict: Comprehensive statistics about zeros and missing values
    """
    # Separate numeric data (excluding country column)
    numeric_df = df.drop('Country', axis=1)
    
    # Analyze missing values (NaN)
    missing_stats = {
        'total_missing': numeric_df.isnull().sum().sum(),
        'missing_percentage': (numeric_df.isnull().sum().sum() / numeric_df.size) * 100,
        'columns_with_missing': numeric_df.columns[numeric_df.isnull().any()].tolist(),
        'countries_with_missing': df['Country'][numeric_df.isnull().any(axis=1)].tolist(),
    }
    
    # Analyze zero values (0)
    zero_stats = {
        'total_zeros': (numeric_df == 0).sum().sum(),
        'zero_percentage': ((numeric_df == 0).sum().sum() / numeric_df.size) * 100,
        'columns_with_zeros': numeric_df.columns[(numeric_df == 0).any()].tolist(),
        'countries_with_zeros': df['Country'][(numeric_df == 0).any(axis=1)].tolist(),
    }
    
    # Analyze near-zero values (values close to 0, e.g., < 0.001)
    near_zero_threshold = 0.001
    near_zero_stats = {
        'total_near_zero': (numeric_df.abs() < near_zero_threshold).sum().sum(),
        'near_zero_percentage': ((numeric_df.abs() < near_zero_threshold).sum().sum() / numeric_df.size) * 100,
        'columns_with_near_zero': numeric_df.columns[(numeric_df.abs() < near_zero_threshold).any()].tolist(),
    }
    
    # Find where zeros and missing values coincide
    zero_and_missing_locations = []
    for idx, row in df.iterrows():
        country = row['Country']
        for col in numeric_df.columns:
            value = row[col]
            if pd.isna(value):
                zero_and_missing_locations.append({
                    'country': country,
                    'sector': col,
                    'type': 'missing',
                    'value': value
                })
            elif value == 0:
                zero_and_missing_locations.append({
                    'country': country,
                    'sector': col,
                    'type': 'zero',
                    'value': value
                })
            elif abs(value) < near_zero_threshold:
                zero_and_missing_locations.append({
                    'country': country,
                    'sector': col,
                    'type': 'near_zero',
                    'value': value
                })
    
    return {
        'missing_stats': missing_stats,
        'zero_stats': zero_stats,
        'near_zero_stats': near_zero_stats,
        'all_issues': zero_and_missing_locations,
        'summary': {
            'total_cells': numeric_df.size,
            'valid_cells': numeric_df.notnull().sum().sum(),
            'problematic_cells': missing_stats['total_missing'] + zero_stats['total_zeros'],
            'problematic_percentage': ((missing_stats['total_missing'] + zero_stats['total_zeros']) / numeric_df.size) * 100
        }
    }

def find_zeros_by_country_sector(df):
    """
    Find all zero values organized by country and sector.
    
    Parameters:
    df (DataFrame): The CEDA data DataFrame
    
    Returns:
    dict: Zeros organized by country and by sector
    """
    numeric_df = df.drop('Country', axis=1)
    
    # Zeros by country
    zeros_by_country = {}
    for idx, row in df.iterrows():
        country = row['Country']
        zero_sectors = []
        for col in numeric_df.columns:
            if row[col] == 0:
                zero_sectors.append(col)
        if zero_sectors:
            zeros_by_country[country] = {
                'count': len(zero_sectors),
                'sectors': zero_sectors,
                'percentage': (len(zero_sectors) / len(numeric_df.columns)) * 100
            }
    
    # Zeros by sector
    zeros_by_sector = {}
    for col in numeric_df.columns:
        zero_countries = df['Country'][df[col] == 0].tolist()
        if zero_countries:
            zeros_by_sector[col] = {
                'count': len(zero_countries),
                'countries': zero_countries,
                'percentage': (len(zero_countries) / len(df)) * 100
            }
    
    return {
        'by_country': zeros_by_country,
        'by_sector': zeros_by_sector
    }

#----------------------------------------------------------------------------------------------

# Test the functions
if __name__ == "__main__":
    # Assuming ceda_data is already loaded
    
    # Test single value lookup
    print("Testing single value lookup:")
    value = lookup_value(ceda_data, 'Afghanistan', '1111A0')
    print(f"Afghanistan, 1111A0: {value}")
    
    # Test sector values across all countries
    print("\nTesting sector values lookup:")
    sector_values = lookup_sector_values(ceda_data, '1111A0')
    if sector_values is not None:
        print(f"1111A0 values for first 5 countries:")
        print(sector_values.head())
    
    # Test country values across all sectors
    print("\nTesting country values lookup:")
    country_values = lookup_country_values(ceda_data, 'Afghanistan')
    if country_values is not None:
        print(f"Afghanistan values for first 10 sectors:")
        print(country_values.head(10))
    
    # Find missing values
    print("\nChecking for missing values:")
    missing_info = find_missing_values(ceda_data)
    print(f"Total missing values: {missing_info['total_missing']}")
    print(f"Missing percentage: {missing_info['missing_percentage']:.2f}%")
    
    if missing_info['total_missing'] > 0:
        print(f"\nMissing values per column (top 10):")
        print(missing_info['missing_per_column'].head(10))
        
        print(f"\nCountries with missing values (top 10):")
        print(missing_info['missing_per_country'].head(10))