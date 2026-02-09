import pandas as pd
import numpy as np

# Read the CSV data
df = pd.read_csv('data/Worldbank_Data.csv')

# Identify the year columns (excluding metadata columns)
year_cols = ['2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]', '2024 [YR2024]']

# Get unique countries and indicators
countries = df['Country Name'].unique()
indicators = df['Series Name'].unique()

# Create a mapping of country name to country code
country_code_mapping = {}
for _, row in df.iterrows():
    country_name = row['Country Name']
    country_code = row['Country Code']
    if country_name not in country_code_mapping:
        country_code_mapping[country_name] = country_code

# Create a dictionary to store the latest values
latest_data = {}

# Process each country and indicator
for country in countries:
    latest_data[country] = {}
    country_data = df[df['Country Name'] == country]
    
    # Get country code for this country
    country_code = country_code_mapping.get(country, '')
    latest_data[country]['Country Code'] = country_code
    
    for indicator in indicators:
        indicator_data = country_data[country_data['Series Name'] == indicator]
        
        if not indicator_data.empty:
            # Get the row for this indicator
            row = indicator_data.iloc[0]
            
            # Find the latest non-null value (going backwards from 2024 to 2020)
            latest_value = None
            
            # Check each year in reverse order
            for year in reversed(year_cols):
                value = row[year]
                # Check if value is not null/NaN and not '..'
                if pd.notna(value) and value != '..':
                    try:
                        # Try to convert to float if possible
                        latest_value = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        latest_value = str(value)
                    break
            
            latest_data[country][indicator] = latest_value
        else:
            latest_data[country][indicator] = None

# Create a DataFrame from the dictionary
result_df = pd.DataFrame.from_dict(latest_data, orient='index')

# Reset index to have Country Name as a column
result_df = result_df.reset_index().rename(columns={'index': 'Country Name'})

# Reorder columns to have Country Name and Country Code first, then all indicators
cols = ['Country Name', 'Country Code'] + list(indicators)
result_df = result_df[cols]

# Save to CSV
result_df.to_csv('country_indicator_latest_values.csv', index=False)

# Display the shape and first few rows
print(f"DataFrame shape: {result_df.shape}")
print(f"\nNumber of countries: {len(countries)}")
print(f"Number of indicators: {len(indicators)}")
print("\nFirst few rows of the result:")
print(result_df[['Country Name', 'Country Code'] + list(indicators)[:5]].head().to_string())

# Display a sample of the data structure
print("\nSample of data for first country:")
sample_country = countries[0]
sample_row = result_df[result_df['Country Name'] == sample_country].iloc[0]
print(f"Country: {sample_row['Country Name']}")
print(f"Country Code: {sample_row['Country Code']}")
for indicator in indicators[:5]:  # Show first 5 indicators
    value = sample_row[indicator]
    print(f"{indicator[:50]:50} : {value}")

# Verify country codes are included
print("\n" + "="*80)
print("VERIFICATION - Sample of country codes:")
print("="*80)
for i, (name, code) in enumerate(country_code_mapping.items()):
    if i < 10:  # Show first 10
        print(f"{name:<30} : {code}")
    else:
        break