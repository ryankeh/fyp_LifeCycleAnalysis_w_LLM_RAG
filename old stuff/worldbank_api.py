import wbgapi as wb
import pandas as pd
import numpy as np

def get_wb_data(countries='all', start_year=2010, end_year=2022):
    """Get World Bank data with corrected indicators"""
    
    # VERIFIED indicators that exist in WB API
    indicators = {
        'EG.ELC.FOSL.ZS': 'fossil_elec',           # Fossil fuel electricity %
        'NY.GDP.PCAP.PP.KD': 'gdp_pc_ppp',         # GDP per capita, PPP
        'NV.IND.MANF.ZS': 'manufacturing_pct',     # Manufacturing, value added % of GDP
        'NE.EXP.GNFS.ZS': 'exports_pct',           # Exports of goods and services % of GDP
        'EG.USE.PCAP.KG.OE': 'energy_use_pc',      # Energy use per capita
        'SP.URB.TOTL.IN.ZS': 'urbanization',       # Urban population % of total
        'EG.ELC.ACCS.ZS': 'elec_access',           # Access to electricity % of population
        'AG.LND.AGRI.ZS': 'agri_land',             # Agricultural land % of land area
    }
    
    # Test one indicator first
    print("Testing indicator availability...")
    test = wb.data.DataFrame('EG.ELC.FOSL.ZS', time=2020, labels=True)
    print(f"Test successful. Found {len(test)} countries")
    
    df_list = []
    
    for code, name in indicators.items():
        print(f"Fetching {code} ({name})...")
        try:
            # Use wb.data.get() for cleaner API call
            data = wb.data.get(code, economy=wb.region.members('WLD'), 
                              time=range(start_year, end_year+1),
                              labels=True, skipBlanks=True)
            
            # Convert to DataFrame
            temp_df = pd.DataFrame(data)
            temp_df.columns = ['country', 'country_code', 'series', 'series_code', 'year', 'value']
            
            # Pivot
            temp_df = temp_df[['country', 'year', 'value']]
            temp_df = temp_df.rename(columns={'value': name})
            temp_df['year'] = temp_df['year'].astype(int)
            
            df_list.append(temp_df)
            print(f"  Found {len(temp_df)} records")
            
        except Exception as e:
            print(f"  Warning: {code} failed - {str(e)[:50]}...")
    
    # Merge all dataframes
    if df_list:
        df = df_list[0]
        for temp_df in df_list[1:]:
            df = pd.merge(df, temp_df, on=['country', 'year'], how='outer')
    else:
        df = pd.DataFrame()
    
    return df

def get_country_metadata():
    """Get country region and income level information"""
    try:
        # Get economies list
        economies = wb.economy.list()
        
        # Convert to DataFrame
        meta_df = pd.DataFrame(economies)
        
        # Different column names in newer wbgapi
        column_map = {}
        if 'id' in meta_df.columns:
            column_map['id'] = 'country_code'
        if 'value' in meta_df.columns:
            column_map['value'] = 'country'
        if 'region' in meta_df.columns:
            # Region is an object, extract ID
            meta_df['region_id'] = meta_df['region'].apply(lambda x: x['id'] if isinstance(x, dict) else x)
            column_map['region_id'] = 'region'
        if 'incomeLevel' in meta_df.columns:
            meta_df['income_id'] = meta_df['incomeLevel'].apply(lambda x: x['id'] if isinstance(x, dict) else x)
            column_map['income_id'] = 'income'
        
        # Rename columns
        meta_df = meta_df.rename(columns=column_map)
        
        # Keep only needed columns
        keep_cols = [c for c in ['country_code', 'country', 'region', 'income'] if c in meta_df.columns]
        meta_df = meta_df[keep_cols]
        
        return meta_df
        
    except Exception as e:
        print(f"Metadata error: {e}")
        # Return simple fallback
        return pd.DataFrame(columns=['country_code', 'country', 'region', 'income'])

def impute_missing_data(df, meta_df):
    """Two-stage imputation with corrected logic"""
    if df.empty or meta_df.empty:
        return df
    
    # Merge with metadata
    df = pd.merge(df, meta_df, on='country', how='left')
    
    # List of indicator columns (exclude metadata)
    indicator_cols = [col for col in df.columns 
                     if col not in ['country', 'year', 'country_code', 'region', 'income']]
    
    print(f"\nImputing missing data for {len(indicator_cols)} indicators...")
    
    # Stage 1: Regional averages (same year)
    for col in indicator_cols:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            # Calculate regional medians by year
            df['regional_median'] = df.groupby(['region', 'year'])[col].transform('median')
            df[col] = df[col].fillna(df['regional_median'])
            missing_after = df[col].isna().sum()
            print(f"  {col}: Regional imputation filled {missing_before - missing_after} values")
    
    # Stage 2: Income group averages
    for col in indicator_cols:
        missing_before = df[col].isna().sum()
        if missing_before > 0:
            # Calculate income group medians by year
            df['income_median'] = df.groupby(['income', 'year'])[col].transform('median')
            df[col] = df[col].fillna(df['income_median'])
            missing_after = df[col].isna().sum()
            print(f"  {col}: Income imputation filled {missing_before - missing_after} values")
    
    # Clean up
    df = df.drop(columns=['regional_median', 'income_median'], errors='ignore')
    
    return df

def main():
    """Main execution function"""
    print("=" * 60)
    print("WORLD BANK DATA COLLECTOR")
    print("=" * 60)
    
    # Step 1: Get country metadata
    print("\n1. Fetching country metadata...")
    meta_df = get_country_metadata()
    print(f"   Found {len(meta_df)} countries")
    
    # Step 2: Get indicator data
    print("\n2. Fetching indicator data (2010-2022)...")
    df = get_wb_data()
    
    if df.empty:
        print("ERROR: No data retrieved. Check your internet connection or API keys.")
        return
    
    print(f"   Initial data shape: {df.shape}")
    print(f"   Countries in data: {df['country'].nunique()}")
    
    # Step 3: Check initial coverage
    print("\n3. Initial data coverage:")
    coverage = df.notna().mean().sort_values(ascending=False)
    for col, cov in coverage.items():
        if col not in ['country', 'year']:
            print(f"   {col:20s}: {cov:.1%}")
    
    # Step 4: Impute missing data
    print("\n4. Imputing missing values...")
    df_imputed = impute_missing_data(df, meta_df)
    
    # Step 5: Final coverage check
    print("\n5. Final data coverage:")
    final_coverage = df_imputed.notna().mean().sort_values(ascending=False)
    for col, cov in final_coverage.items():
        if col not in ['country', 'year', 'country_code', 'region', 'income']:
            print(f"   {col:20s}: {cov:.1%}")
    
    # Step 6: Check Afghanistan specifically
    print("\n6. Afghanistan data sample:")
    afg_data = df_imputed[df_imputed['country'] == 'Afghanistan']
    if not afg_data.empty:
        # Get most recent year
        latest = afg_data.sort_values('year', ascending=False).iloc[0]
        for col in ['fossil_elec', 'gdp_pc_ppp', 'manufacturing_pct', 'urbanization']:
            if col in latest:
                print(f"   {col:20s}: {latest[col]:.2f}")
    else:
        print("   No data for Afghanistan found")
    
    # Step 7: Save to CSV
    print("\n7. Saving data to CSV...")
    df_imputed.to_csv('worldbank_indicators.csv', index=False)
    meta_df.to_csv('country_metadata.csv', index=False)
    print("   Saved: worldbank_indicators.csv")
    print("   Saved: country_metadata.csv")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()