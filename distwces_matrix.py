import pandas as pd
import numpy as np

# Load the distance dataset
dist_df = pd.read_excel('data/dist_cepii.xls', sheet_name='dist_cepii')

# Load the countries from your final dataset
countries_df = pd.read_csv('countries_in_final_dataset.csv')
final_countries = sorted(countries_df['Country Code'].tolist())

print(f"Countries in final dataset: {len(final_countries)}")
print(f"First 20: {final_countries[:20]}")

print(f"\nOriginal dataset shape: {dist_df.shape}")

# ============================================================================
# FILTER DISTANCE DATA TO ONLY INCLUDE FINAL COUNTRIES
# ============================================================================

# Filter to only include rows where both origin AND destination are in final countries
filtered_dist_df = dist_df[
    dist_df['iso_o'].isin(final_countries) & 
    dist_df['iso_d'].isin(final_countries)
].copy()

print(f"Filtered dataset shape: {filtered_dist_df.shape}")
print(f"Rows after filtering: {len(filtered_dist_df):,}")

# ============================================================================
# CREATE COUNTRY × COUNTRY MATRIX (ONLY FOR FINAL COUNTRIES)
# ============================================================================

# Initialize an empty matrix with float dtype
n_countries = len(final_countries)
dist_matrix = pd.DataFrame(
    index=final_countries,
    columns=final_countries,
    data=np.nan,
    dtype=float
)

# Fill the matrix with distances from the filtered dataset
successful_conversions = 0
failed_conversions = 0

for _, row in filtered_dist_df.iterrows():
    origin = row['iso_o']
    destination = row['iso_d']
    distance_value = row['distwces']
    
    # Try to convert to float, handle errors
    try:
        distance = float(distance_value)
        dist_matrix.loc[origin, destination] = distance
        successful_conversions += 1
    except (ValueError, TypeError):
        failed_conversions += 1

print(f"Conversions: {successful_conversions} successful, {failed_conversions} failed")

# ============================================================================
# CHECK COMPLETENESS
# ============================================================================

total_cells = n_countries * n_countries
filled_cells = dist_matrix.count().sum()
missing_cells = total_cells - filled_cells

print(f"\nMatrix dimensions: {n_countries} × {n_countries}")
print(f"Total possible pairs: {total_cells:,}")
print(f"Pairs with distance data: {filled_cells:,}")
print(f"Coverage: {filled_cells/total_cells*100:.2f}%")

# ============================================================================
# SAVE THE MATRIX
# ============================================================================

# Save as CSV
output_file = 'country_distance_matrix_distwces.csv'
dist_matrix.to_csv(output_file, float_format='%.6f')

print(f"\n✅ Country × country matrix saved to: {output_file}")