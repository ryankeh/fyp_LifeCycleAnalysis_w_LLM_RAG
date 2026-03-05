import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the CSV file
file_path = 'data/WDICSV.csv'
df = pd.read_csv(file_path)

print(f"Shape of dataframe: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Number of unique countries: {df['Country Name'].nunique()}")
print(f"Number of unique indicators: {df['Indicator Name'].nunique()}")

# Define the 8 confirmed indicators with updated mapping
# Replaced 'Fossil fuel energy consumption' with 'Electricity production from coal sources (% of total)'
indicators = {
    'GDP per capita, PPP (constant 2021 international $)': 'NY.GDP.PCAP.PP.KD',
    'Industry (including construction), value added (% of GDP)': 'NV.IND.TOTL.ZS',
    'Renewable energy consumption (% of total final energy consumption)': 'EG.FEC.RNEW.ZS',
    'Electricity production from coal sources (% of total)': 'EG.ELC.COAL.ZS',  # Replacement indicator
    'Energy intensity level of primary energy (MJ/$2021 PPP GDP)': 'EG.EGY.PRIM.PP.KD',
    'GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)': 'EG.GDP.PUSE.KO.PP.KD',
    'Urban population (% of total population)': 'SP.URB.TOTL.IN.ZS',
    'Total natural resources rents (% of GDP)': 'NY.GDP.TOTL.RT.ZS'
}

# Filter the dataframe to only include these indicators
df_filtered = df[df['Indicator Code'].isin(indicators.values())].copy()

if df_filtered.empty:
    print("\nNo matching indicators found. Available indicators in the file:")
    print(df['Indicator Name'].unique()[:20])
else:
    print(f"\nFound {df_filtered['Indicator Name'].nunique()} out of 8 indicators")
    print("\nIndicators found:")
    for name in df_filtered['Indicator Name'].unique():
        code = df_filtered[df_filtered['Indicator Name'] == name]['Indicator Code'].iloc[0]
        print(f"  - {name} ({code})")

# Function to get the most recent non-null value and its year for each country-indicator pair
def get_latest_value_and_year(row):
    # Years are from column 5 onwards (1960 onwards in CSV format)
    # In the CSV, columns might be named differently - let's find all year columns
    year_cols = [col for col in df.columns if col.isdigit() or (col.startswith('19') or col.startswith('20'))]
    
    if not year_cols:
        # If no year columns found, assume all columns after the first 4 are years
        year_cols = df.columns[4:].tolist()
    
    values = []
    years = []
    
    for year in year_cols:
        if year in row.index:
            val = row[year]
            if pd.notna(val) and val != '':
                values.append(val)
                years.append(int(year))
    
    if values:
        # Find the most recent year with data
        latest_idx = np.argmax(years)
        return pd.Series({
            'value': values[latest_idx],
            'year': years[latest_idx]
        })
    
    return pd.Series({'value': np.nan, 'year': np.nan})

# Create the two output dataframes
if not df_filtered.empty:
    # Pivot to get countries as rows, indicators as columns with latest values
    latest_values = []
    latest_years = []
    
    # Get unique countries
    countries = df_filtered['Country Name'].unique()
    print(f"\nProcessing data for {len(countries)} countries...")
    
    for country in countries:
        country_data = {'Country Name': country, 'Country Code': None}
        year_data = {'Country Name': country, 'Country Code': None}
        
        country_rows = df_filtered[df_filtered['Country Name'] == country]
        
        # Get country code
        if not country_rows.empty:
            country_data['Country Code'] = country_rows.iloc[0]['Country Code']
            year_data['Country Code'] = country_rows.iloc[0]['Country Code']
        
        for indicator_name, indicator_code in indicators.items():
            # Get the row for this indicator
            indicator_row = country_rows[country_rows['Indicator Code'] == indicator_code]
            
            if not indicator_row.empty:
                # Get latest value and year
                result = get_latest_value_and_year(indicator_row.iloc[0])
                country_data[indicator_name] = result['value']
                year_data[indicator_name] = result['year']
            else:
                country_data[indicator_name] = np.nan
                year_data[indicator_name] = np.nan
        
        latest_values.append(country_data)
        latest_years.append(year_data)
    
    # Create dataframes
    df_values = pd.DataFrame(latest_values)
    df_years = pd.DataFrame(latest_years)
    
    # Sort by country name
    df_values = df_values.sort_values('Country Name').reset_index(drop=True)
    df_years = df_years.sort_values('Country Name').reset_index(drop=True)
    
    # Save to Excel files
    output_file_values = 'carbon_intensity_indicators_values.xlsx'
    output_file_years = 'carbon_intensity_indicators_years.xlsx'
    
    with pd.ExcelWriter(output_file_values, engine='openpyxl') as writer:
        df_values.to_excel(writer, sheet_name='Indicator Values', index=False)
    
    with pd.ExcelWriter(output_file_years, engine='openpyxl') as writer:
        df_years.to_excel(writer, sheet_name='Data Years', index=False)
    
    print(f"\nFiles saved successfully:")
    print(f"1. {output_file_values}")
    print(f"2. {output_file_years}")
    
    # ===== VISUALIZATION 1: Data Coverage =====
    plt.figure(figsize=(14, 8))
    
    # Calculate coverage for each indicator
    coverage_data = []
    for indicator in indicators.keys():
        if indicator in df_values.columns:
            non_null = df_values[indicator].notna().sum()
            total = len(df_values)
            coverage_pct = (non_null / total) * 100
            coverage_data.append({
                'Indicator': indicator[:50] + '...' if len(indicator) > 50 else indicator,
                'Coverage (%)': coverage_pct,
                'Countries with Data': non_null
            })
    
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('Coverage (%)', ascending=True)
    
    # Create horizontal bar chart
    colors = plt.cm.YlOrRd(coverage_df['Coverage (%)'] / 100)
    bars = plt.barh(coverage_df['Indicator'], coverage_df['Coverage (%)'], color=colors)
    
    # Add value labels
    for i, (bar, pct, count) in enumerate(zip(bars, coverage_df['Coverage (%)'], coverage_df['Countries with Data'])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}% ({count} countries)', 
                va='center', fontsize=10)
    
    plt.xlabel('Coverage (% of countries)', fontsize=12)
    plt.title('WDI Indicator Coverage Across Countries', fontsize=16, fontweight='bold')
    plt.xlim(0, 105)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save coverage plot
    plt.savefig('indicator_coverage.png', dpi=300, bbox_inches='tight')
    print("3. indicator_coverage.png - Coverage visualization saved")
    
    # ===== VISUALIZATION 2: Latest Year Distribution =====
    plt.figure(figsize=(16, 10))
    
    # Prepare data for year distribution
    year_data_long = []
    for indicator in indicators.keys():
        if indicator in df_years.columns:
            years = df_years[indicator].dropna()
            for year in years:
                year_data_long.append({
                    'Indicator': indicator[:50] + '...' if len(indicator) > 50 else indicator,
                    'Year': year
                })
    
    year_df = pd.DataFrame(year_data_long)
    
    # Create box plot of year distribution by indicator
    if not year_df.empty:
        # Sort indicators by median year
        median_years = year_df.groupby('Indicator')['Year'].median().sort_values()
        year_df['Indicator'] = pd.Categorical(year_df['Indicator'], categories=median_years.index, ordered=True)
        
        # Create violin plot with box plot inside
        plt.subplot(2, 1, 1)
        sns.violinplot(data=year_df, x='Indicator', y='Year', inner='box', cut=0)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylabel('Year', fontsize=12)
        plt.title('Distribution of Latest Data Years by Indicator', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add a horizontal line for the current year
        current_year = datetime.now().year
        plt.axhline(y=current_year, color='red', linestyle='--', alpha=0.5, label=f'Current Year ({current_year})')
        plt.legend()
        
        # Create heatmap of data availability by year
        plt.subplot(2, 1, 2)
        
        # Create a matrix of data availability
        indicators_list = list(indicators.keys())
        years_range = range(1990, current_year + 1)
        
        availability_matrix = []
        y_labels = []
        
        for indicator in indicators_list:
            if indicator in df_years.columns:
                y_labels.append(indicator[:40] + '...' if len(indicator) > 40 else indicator)
                indicator_availability = []
                ind_code = indicators[indicator]
                
                for year in years_range:
                    # Count countries with data in this year for this indicator
                    count = 0
                    for _, row in df_filtered[df_filtered['Indicator Code'] == ind_code].iterrows():
                        if str(year) in row.index and pd.notna(row[str(year)]) and row[str(year)] != '':
                            count += 1
                    indicator_availability.append(count)
                
                availability_matrix.append(indicator_availability)
        
        if availability_matrix:
            # Normalize for better visualization
            availability_matrix = np.array(availability_matrix)
            
            # Create heatmap
            im = plt.imshow(availability_matrix, aspect='auto', cmap='YlGnBu', 
                           interpolation='nearest', vmin=0)
            
            # Customize axes
            plt.yticks(range(len(y_labels)), y_labels, fontsize=10)
            plt.xticks(range(0, len(years_range), 5), 
                      [years_range[i] for i in range(0, len(years_range), 5)], 
                      rotation=45, ha='right')
            
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Indicator', fontsize=12)
            plt.title('Data Availability Heatmap (Number of Countries with Data)', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Number of Countries', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('indicator_year_distribution.png', dpi=300, bbox_inches='tight')
    print("4. indicator_year_distribution.png - Year distribution visualization saved")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nData Coverage by Indicator:")
    for idx, row in coverage_df.iterrows():
        print(f"  {row['Indicator']}: {row['Coverage (%)']:.1f}% ({row['Countries with Data']} countries)")
    
    print("\nLatest Year Statistics by Indicator:")
    for indicator in indicators.keys():
        if indicator in df_years.columns:
            years_data = df_years[indicator].dropna()
            if not years_data.empty:
                print(f"\n  {indicator[:60]}:")
                print(f"    Median year: {years_data.median():.0f}")
                print(f"    Range: {years_data.min():.0f} - {years_data.max():.0f}")
                print(f"    Most common year: {years_data.mode().iloc[0] if not years_data.mode().empty else 'N/A'}")
    
    # Show first few rows as preview
    print("\n" + "="*60)
    print("PREVIEW OF OUTPUT FILES")
    print("="*60)
    print("\nFirst 5 rows (Values):")
    print(df_values[['Country Name'] + list(indicators.keys())].head())
    print("\nFirst 5 rows (Years):")
    print(df_years[['Country Name'] + list(indicators.keys())].head())
    
else:
    print("\nNo matching indicators found. Creating a list of available indicators for your reference...")
    
    # Get unique indicator names and their codes
    available_indicators = df[['Indicator Name', 'Indicator Code']].drop_duplicates().sort_values('Indicator Name')
    
    # Save to Excel for reference
    available_indicators.to_excel('available_wdi_indicators.xlsx', index=False)
    print("Saved list of all available indicators to 'available_wdi_indicators.xlsx'")
    
    # Show a sample of available indicators
    print("\nSample of available indicators (first 20):")
    print(available_indicators.head(20).to_string(index=False))