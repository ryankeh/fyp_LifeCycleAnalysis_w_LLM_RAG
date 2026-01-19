import numpy as np
import pandas as pd
import random

def convert_zeros_to_true_nulls(df):
    """
    Convert all 0 values to NaN (true nulls from database).
    
    Parameters:
    df (DataFrame): Original CEDA data with 'Country' column
    
    Returns:
    tuple: (true_null_df, conversion_stats)
        - true_null_df: DataFrame with 0s converted to NaN
        - conversion_stats: Dictionary with statistics about the conversion
    """
    print("Converting 0s to NaN (true database nulls)...")
    
    # Create a copy of the original data
    original_df = df.copy()
    
    # Separate country column
    country_col = original_df['Country']
    numeric_df = original_df.drop('Country', axis=1)
    
    # Count zeros before conversion
    total_cells = numeric_df.size
    zero_counts_per_column = (numeric_df == 0).sum()
    zero_counts_per_country = (numeric_df == 0).sum(axis=1)
    total_zeros = zero_counts_per_column.sum()
    zero_percentage = (total_zeros / total_cells) * 100
    
    # Convert 0s to NaN (true nulls)
    true_null_numeric = numeric_df.replace(0, np.nan)
    true_null_df = pd.concat([country_col, true_null_numeric], axis=1)
    
    # Count NaN after conversion
    nan_counts_per_column = true_null_numeric.isnull().sum()
    nan_counts_per_country = true_null_numeric.isnull().sum(axis=1)
    total_nan = nan_counts_per_column.sum()
    
    # Create statistics
    conversion_stats = {
        'total_cells': total_cells,
        'total_zeros_before': total_zeros,
        'zero_percentage_before': zero_percentage,
        'total_nulls_after': total_nan,
        'null_percentage_after': (total_nan / total_cells) * 100,
        'zero_counts_per_column': zero_counts_per_column,
        'zero_counts_per_country': zero_counts_per_country,
        'nan_counts_per_column': nan_counts_per_column,
        'nan_counts_per_country': nan_counts_per_country,
        'affected_columns': zero_counts_per_column[zero_counts_per_column > 0].index.tolist(),
        'affected_countries': zero_counts_per_country[zero_counts_per_country > 0].index.tolist()
    }
    
    print(f"  Converted {total_zeros:,} zeros to NaN")
    print(f"  Original zeros: {zero_percentage:.2f}% of cells")
    print(f"  New nulls: {(total_nan/total_cells)*100:.2f}% of cells")
    
    return true_null_df, conversion_stats


def create_engineered_nulls(df, null_percentage=0.1, random_seed=42, exclude_columns=None):
    """
    Create engineered nulls by randomly masking known non-null values.
    
    Parameters:
    df (DataFrame): DataFrame (preferably after converting 0s to NaN)
    null_percentage (float): Percentage of non-null values to mask (0-1)
    random_seed (int): Random seed for reproducibility
    exclude_columns (list): Columns to exclude from masking (e.g., ['Country'])
    
    Returns:
    tuple: (masked_df, mask_info)
        - masked_df: DataFrame with engineered nulls added
        - mask_info: Dictionary with information about masked values
    """
    print(f"Creating engineered nulls ({null_percentage*100:.1f}% of non-null values)...")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a copy of the input data
    working_df = df.copy()
    
    # Default exclude columns
    if exclude_columns is None:
        exclude_columns = ['Country']
    
    # Identify columns to mask (numeric columns not in exclude_columns)
    mask_columns = [col for col in working_df.columns 
                   if col not in exclude_columns and working_df[col].dtype in ['float64', 'int64']]
    
    if not mask_columns:
        print("  No numeric columns found for masking!")
        return working_df, {}
    
    # Get positions of non-null (known) values
    known_values_mask = working_df[mask_columns].notnull()
    known_positions = list(zip(*np.where(known_values_mask)))
    
    if not known_positions:
        print("  No known values found to create engineered nulls!")
        return working_df, {}
    
    # Calculate how many values to mask
    num_to_mask = int(len(known_positions) * null_percentage)
    
    # Randomly select positions to mask
    mask_positions = random.sample(known_positions, num_to_mask)
    
    # Create engineered mask and store original values
    engineered_mask = np.zeros_like(working_df[mask_columns], dtype=bool)
    original_values = []
    
    # Apply engineered nulls
    for i, j in mask_positions:
        row_idx = i
        col_name = mask_columns[j]
        original_value = working_df.loc[row_idx, col_name]
        
        # Store original value information
        original_values.append({
            'row_index': row_idx,
            'column_name': col_name,
            'country': working_df.loc[row_idx, 'Country'] if 'Country' in working_df.columns else None,
            'original_value': original_value
        })
        
        # Mask as engineered null
        working_df.loc[row_idx, col_name] = np.nan
        engineered_mask[i, j] = True
    
    # Calculate statistics
    engineered_nan_count = engineered_mask.sum()
    total_nan_after = working_df[mask_columns].isnull().sum().sum()
    original_nan_count = total_nan_after - engineered_nan_count
    
    # Create mask information
    mask_info = {
        'null_percentage': null_percentage,
        'random_seed': random_seed,
        'mask_columns': mask_columns,
        'engineered_nulls_count': engineered_nan_count,
        'original_nulls_count': original_nan_count,
        'total_nulls': total_nan_after,
        'num_known_values': len(known_positions),
        'num_masked_values': num_to_mask,
        'engineered_mask': engineered_mask,
        'mask_positions': mask_positions,
        'original_values': original_values
    }
    
    print(f"  Total known values: {len(known_positions):,}")
    print(f"  Added {engineered_nan_count:,} engineered nulls")
    print(f"  Original nulls: {original_nan_count:,}")
    print(f"  Total nulls after masking: {total_nan_after:,}")
    print(f"  Percentage of engineered nulls: {(engineered_nan_count/total_nan_after*100):.1f}%")
    
    return working_df, mask_info


def get_masked_values_df(mask_info, original_df):
    """
    Create a DataFrame showing all masked values with their original values.
    
    Parameters:
    mask_info (dict): Mask information from create_engineered_nulls
    original_df (DataFrame): Original DataFrame before masking
    
    Returns:
    DataFrame: DataFrame with details of masked values
    """
    if 'original_values' not in mask_info:
        return pd.DataFrame()
    
    masked_data = []
    for info in mask_info['original_values']:
        row_idx = info['row_index']
        col_name = info['column_name']
        
        masked_data.append({
            'Country': info['country'],
            'Sector': col_name,
            'Original_Value': info['original_value'],
            'Row_Index': row_idx,
            'Column_Name': col_name
        })
    
    return pd.DataFrame(masked_data)


# Example usage with both steps separated:
if __name__ == "__main__":
    # Assuming ceda_data is your original DataFrame
    
    print("=" * 60)
    print("STEP 1: Convert 0s to True Nulls (Database Missing Values)")
    print("=" * 60)
    
    # Step 1: Convert 0s to NaN (true nulls)
    true_null_df, true_null_stats = convert_zeros_to_true_nulls(ceda_data)
    
    # Save the true nulls data
    true_null_df.to_csv('ceda_true_nulls.csv', index=False)
    print("Saved true nulls to: ceda_true_nulls.csv")
    
    print("\n" + "=" * 60)
    print("STEP 2: Create Engineered Nulls for Benchmarking")
    print("=" * 60)
    
    # Step 2: Create engineered nulls (10% of known values)
    benchmark_df, mask_info = create_engineered_nulls(
        true_null_df, 
        null_percentage=0.10,
        random_seed=42,
        exclude_columns=['Country']
    )
    
    # Save the benchmark data
    benchmark_df.to_csv('ceda_benchmark.csv', index=False)
    print("Saved benchmark data to: ceda_benchmark.csv")
    
    # Create and save masked values reference
    masked_values_df = get_masked_values_df(mask_info, true_null_df)
    if not masked_values_df.empty:
        masked_values_df.to_csv('masked_values_reference.csv', index=False)
        print("Saved masked values reference to: masked_values_reference.csv")
    
    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print("1. ceda_true_nulls.csv - 0s converted to NaN (true database nulls)")
    print("2. ceda_benchmark.csv - With additional 10% engineered nulls")
    print("3. masked_values_reference.csv - Answer key for engineered nulls")