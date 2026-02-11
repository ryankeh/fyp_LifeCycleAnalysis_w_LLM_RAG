import pandas as pd
import numpy as np

def create_country_bands_df(covariate_df, banding_columns=None, n_bands=4, band_labels=None,
                           min_non_null=20):
    """
    Create a DataFrame with banded covariates (1-4) for hot-deck matching.
    
    Parameters:
    -----------
    covariate_df : DataFrame
        DataFrame containing country covariates with 'Country Code' and 'Country Name'
    banding_columns : list, optional
        List of column names to create bands for.
        If None, bands ALL numeric columns except 'Country Code'.
    n_bands : int
        Number of bands to create (default: 4 for quartiles)
    band_labels : list, optional
        Labels for the bands. If None, uses 1, 2, 3, 4.
    min_non_null : int
        Minimum number of non-null values required to create bands
        
    Returns:
    --------
    DataFrame with same columns as input, but numeric values replaced with band numbers 1-4
    """
    
    # Validate required columns
    required_cols = ['Country Name', 'Country Code']
    for col in required_cols:
        if col not in covariate_df.columns:
            raise ValueError(f"Covariate DataFrame missing '{col}' column")
    
    # Create a copy to avoid modifying original
    bands_df = covariate_df.copy()
    
    # Default banding columns: ALL numeric columns except Country Code
    if banding_columns is None:
        numeric_cols = covariate_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove Country Code if it's numeric (unlikely but safe)
        if 'Country Code' in numeric_cols:
            numeric_cols.remove('Country Code')
        banding_columns = numeric_cols  # NO LIMIT - use ALL numeric columns
    
    print(f"Found {len(banding_columns)} numeric covariates to potentially band:")
    for col in banding_columns:
        non_null = bands_df[col].notna().sum()
        print(f"  {col}: {non_null} non-null values")
    
    print(f"\nCreating bands (1-{n_bands}) for covariates with ≥{min_non_null} non-null values:")
    
    # Default band labels: 1, 2, 3, 4
    if band_labels is None:
        band_labels = list(range(1, n_bands + 1))
    elif len(band_labels) != n_bands:
        raise ValueError(f"band_labels must have {n_bands} elements")
    
    # Track banding statistics
    banding_stats = {}
    successful_bands = []
    failed_bands = []
    
    # Apply banding to each column
    for col in banding_columns:
        if col not in bands_df.columns:
            print(f"  Warning: Column '{col}' not found, skipping")
            failed_bands.append((col, "Column not found"))
            continue
        
        if col in ['Country Name', 'Country Code']:
            continue  # Skip non-covariate columns
        
        # Check if column has enough non-NaN values
        non_null_count = bands_df[col].notna().sum()
        
        if non_null_count < min_non_null:
            print(f"  {col}: Not enough data ({non_null_count} non-null values, need {min_non_null}), skipping")
            # Keep original values instead of NaN for unbanded columns
            failed_bands.append((col, f"Insufficient data ({non_null_count} non-null)"))
            continue
        
        try:
            # Create a temporary series for banding
            temp_series = bands_df[col].copy()
            
            # Get indices of non-null values
            non_null_idx = temp_series.dropna().index
            
            if len(non_null_idx) == 0:
                print(f"  {col}: No valid data after cleaning, skipping")
                failed_bands.append((col, "No valid data after cleaning"))
                continue
            
            # Apply qcut only to non-null values
            temp_series_non_null = temp_series.loc[non_null_idx]
            
            # FIX: Use rank-based quantiles instead of pd.qcut to handle duplicates
            try:
                # First try pd.qcut
                bands_categorical, bins = pd.qcut(
                    temp_series_non_null,
                    q=n_bands,
                    labels=band_labels,
                    retbins=True,
                    duplicates='drop'
                )
                
                # Check if we got the right number of bins
                if len(bins) != n_bands + 1:
                    # Use custom binning based on percentiles
                    percentiles = [i * 100 / n_bands for i in range(n_bands + 1)]
                    bins = np.percentile(temp_series_non_null, percentiles)
                    # Make sure bins are unique
                    bins = np.unique(bins)
                    
                    if len(bins) < 2:
                        raise ValueError("Not enough unique values for binning")
                    
                    # Create bands using cut with custom bins
                    bands_categorical = pd.cut(
                        temp_series_non_null,
                        bins=bins,
                        labels=band_labels[:len(bins)-1],
                        include_lowest=True
                    )
            
            except Exception as qcut_error:
                # Use percentile-based binning as fallback
                print(f"    pd.qcut failed, using percentile-based binning: {qcut_error}")
                percentiles = [i * 100 / n_bands for i in range(n_bands + 1)]
                bins = np.percentile(temp_series_non_null, percentiles)
                bins = np.unique(bins)  # Remove duplicates
                
                if len(bins) < 2:
                    raise ValueError(f"Only {len(bins)} unique bins after de-duplication")
                
                bands_categorical = pd.cut(
                    temp_series_non_null,
                    bins=bins,
                    labels=band_labels[:len(bins)-1],
                    include_lowest=True
                )
            
            # Convert categorical bands to integers (1-4)
            # FIX: Convert to string first, then to integer to avoid float conversion
            bands_int = bands_categorical.astype(str).str.extract('(\d+)')[0].astype(int)
            
            # Update the original series
            temp_series.loc[non_null_idx] = bands_int
            
            # Assign back to dataframe with proper integer type
            bands_df[col] = temp_series.astype('Int64')  # Note: capital 'I' for nullable integer
            
            # Store statistics
            band_counts = bands_df[col].value_counts().sort_index().to_dict()
            
            # Clean up band_counts dictionary (convert Int64 to regular int)
            clean_band_counts = {}
            for k, v in band_counts.items():
                if pd.notna(k):
                    clean_band_counts[int(k)] = int(v)
            
            missing_count = bands_df[col].isna().sum()
            
            # Store bins as list of floats
            if 'bins' in locals():
                bins_list = [float(b) for b in bins]
            else:
                bins_list = []
            
            banding_stats[col] = {
                'bins': bins_list,
                'band_counts': clean_band_counts,
                'missing': int(missing_count),
                'non_null_count': non_null_count,
                'dtype': str(bands_df[col].dtype)
            }
            
            successful_bands.append(col)
            print(f"  {col}: ✓ Successfully banded")
            print(f"     Bins: {[round(b, 2) for b in bins_list]}")
            print(f"     Distribution: {clean_band_counts}")
            print(f"     Missing: {missing_count}")
            print(f"     Data type: {bands_df[col].dtype}")
            
        except Exception as e:
            print(f"  {col}: ✗ Banding failed - {str(e)}")
            # Keep original values for failed banding
            failed_bands.append((col, str(e)))
    
    # Print summary
    print(f"\n{'='*60}")
    print("BANDING SUMMARY")
    print(f"{'='*60}")
    print(f"Total countries: {len(bands_df)}")
    print(f"Successfully banded: {len(successful_bands)} covariates")
    print(f"Failed to band: {len(failed_bands)} covariates")
    
    if successful_bands:
        print(f"\nSuccessfully banded columns:")
        for col in successful_bands:
            stats = banding_stats[col]
            if stats['bins']:
                print(f"  {col}: Bins={[round(b, 2) for b in stats['bins']]}")
            else:
                print(f"  {col}: Custom percentile bins")
    
    if failed_bands:
        print(f"\nFailed columns:")
        for col, reason in failed_bands:
            print(f"  {col}: {reason}")
    
    # Calculate completeness
    if successful_bands:
        completeness_df = bands_df[successful_bands].notna()
        completeness = completeness_df.mean().mean() * 100
        print(f"\nAverage completeness across banded columns: {completeness:.1f}%")
        
        # Show completeness per column
        print(f"\nCompleteness per column:")
        for col in successful_bands:
            comp = bands_df[col].notna().mean() * 100
            print(f"  {col}: {comp:.1f}% complete")
    
    # Save to Excel
    excel_filename = 'country_bands.xlsx'
    
    # Create a writer object
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Save banded data
        bands_df.to_excel(writer, sheet_name='Banded_Data', index=False)
        
        # Save banding statistics
        if banding_stats:
            stats_data = []
            for col, stats in banding_stats.items():
                row = {
                    'Column': col,
                    'Data_Type': stats['dtype'],
                    'Non_Null_Count': stats['non_null_count'],
                    'Missing_Count': stats['missing'],
                    'Completeness': f"{(stats['non_null_count'] / (stats['non_null_count'] + stats['missing'])) * 100:.1f}%"
                }
                
                # Add bin information
                if stats['bins'] and len(stats['bins']) > 1:
                    for i in range(len(stats['bins']) - 1):
                        row[f'Bin_{i+1}_Range'] = f"[{stats['bins'][i]:.2f}, {stats['bins'][i+1]:.2f}]"
                
                # Add band counts
                for band in range(1, n_bands + 1):
                    row[f'Band_{band}_Count'] = stats['band_counts'].get(band, 0)
                
                stats_data.append(row)
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Banding_Statistics', index=False)
        
        # Save failed bands info
        if failed_bands:
            failed_df = pd.DataFrame(failed_bands, columns=['Column', 'Failure_Reason'])
            failed_df.to_excel(writer, sheet_name='Failed_Banding', index=False)
        
        # Save data dictionary
        dict_data = []
        for col in bands_df.columns:
            dict_data.append({
                'Column_Name': col,
                'Data_Type': str(bands_df[col].dtype),
                'Description': 'Banded covariate (1-4)' if col in successful_bands else 
                              ('Original value' if col in banding_columns else 'Identifier'),
                'Non_Null_Count': bands_df[col].notna().sum(),
                'Missing_Count': bands_df[col].isna().sum()
            })
        
        dict_df = pd.DataFrame(dict_data)
        dict_df.to_excel(writer, sheet_name='Data_Dictionary', index=False)
    
    print(f"\nBanded data saved to: {excel_filename}")
    print(f"  - Sheet 'Banded_Data': Banded covariates (1-4 values)")
    print(f"  - Sheet 'Banding_Statistics': Bin ranges and distributions")
    print(f"  - Sheet 'Data_Dictionary': Column information")
    if failed_bands:
        print(f"  - Sheet 'Failed_Banding': Columns that couldn't be banded")
    
    # Verify integer types
    print(f"\nVerifying data types:")
    int_columns = []
    for col in successful_bands:
        if pd.api.types.is_integer_dtype(bands_df[col]):
            int_columns.append(col)
            print(f"  {col}: ✓ Integer type ({bands_df[col].dtype})")
        else:
            print(f"  {col}: ✗ Not integer type ({bands_df[col].dtype})")
    
    print(f"\nTotal integer banded columns: {len(int_columns)}")
    
    return bands_df










def hotdeck_imputation_multiple_targets(target_df, bands_df, target_columns=None, 
                                       match_vars=None, random_state=42, 
                                       donor_threshold=1, verbose=True):
    """
    Perform hot-deck imputation using the banded covariates DataFrame.
    
    Parameters:
    -----------
    target_df : DataFrame
        DataFrame with missing values to impute.
        Must have 'Country' and 'Country Code' columns.
    bands_df : DataFrame
        DataFrame from create_country_bands_df() with banded covariates (1-4 values).
    target_columns : list, optional
        List of columns to impute. If None, imputes all numeric columns
        except 'Country' and 'Country Code'.
    match_vars : list, optional
        List of banded covariate columns to use for matching.
        If None, uses all integer-type banded columns.
    random_state : int
        Random seed for reproducibility.
    donor_threshold : int
        Minimum matching donors required.
    verbose : bool
        Whether to print progress information.
        
    Returns:
    --------
    imputed_df : DataFrame with imputed values
    imputation_log : Dict with detailed imputation records
    """
    
    # Validate inputs
    required_target_cols = ['Country', 'Country Code']
    for col in required_target_cols:
        if col not in target_df.columns:
            raise ValueError(f"Target DataFrame missing '{col}' column")
    
    required_band_cols = ['Country Code']
    for col in required_band_cols:
        if col not in bands_df.columns:
            raise ValueError(f"Bands DataFrame missing '{col}' column")
    
    # Identify target columns
    if target_columns is None:
        numeric_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [col for col in numeric_cols if col not in ['Country', 'Country Code']]
    
    if verbose:
        print(f"Target columns to impute: {target_columns}")
        print(f"Number of countries in target data: {len(target_df)}")
        print(f"Number of countries in bands data: {len(bands_df)}")
    
    # Merge target data with banded covariates
    merged_data = pd.merge(
        target_df[['Country', 'Country Code'] + target_columns],
        bands_df,
        on='Country Code',
        how='left'
    )
    
    if verbose:
        print(f"Merged data shape: {merged_data.shape}")
        print(f"Missing values in merged data: {merged_data[target_columns].isna().sum().sum()}")
    
    # Identify which covariates to use for matching
    if match_vars is None:
        # Use all integer-type banded columns (1-4 values)
        match_vars = []
        for col in bands_df.columns:
            if col not in ['Country Name', 'Country Code']:
                # Check if it's integer type (banded column)
                if pd.api.types.is_integer_dtype(bands_df[col]):
                    match_vars.append(col)
                # Also include float columns that might be banded (1.0, 2.0, etc.)
                elif pd.api.types.is_float_dtype(bands_df[col]) and bands_df[col].dropna().isin([1.0, 2.0, 3.0, 4.0]).all():
                    match_vars.append(col)
    
    if verbose:
        print(f"\nUsing {len(match_vars)} banded covariates for matching:")
        for var in match_vars[:10]:  # Show first 10
            unique_vals = merged_data[var].dropna().unique()
            print(f"  {var}: {len(unique_vals)} unique values (range: {sorted(unique_vals)})")
        if len(match_vars) > 10:
            print(f"  ... and {len(match_vars) - 10} more")
    
    # Initialize results
    imputed_data = target_df.copy()
    imputation_log = {}
    np.random.seed(random_state)
    
    # Process each target column
    for target_col in target_columns:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing: {target_col}")
            print(f"{'='*50}")
        
        # Identify missing values for this column
        missing_mask = merged_data[target_col].isna()
        
        if missing_mask.sum() == 0:
            if verbose:
                print(f"No missing values in {target_col}")
            continue
        
        recipients = merged_data[missing_mask].copy()
        donors = merged_data[~missing_mask].copy()
        
        if verbose:
            print(f"Missing values: {len(recipients)}")
            print(f"Available donors: {len(donors)}")
        
        col_log = []
        donor_usage = {}
        fallback_counts = {'exact': 0, 'partial': 0, 'band_mean': 0, 'global_mean': 0}
        
        # Process each recipient
        for idx, recipient in recipients.iterrows():
            recipient_code = recipient['Country Code']
            recipient_name = recipient['Country']
            
            # Find exact matching donors based on ALL match variables
            matching_donors = donors.copy()
            exact_match = True
            
            for match_var in match_vars:
                if match_var in recipient and pd.notna(recipient[match_var]):
                    matching_donors = matching_donors[
                        matching_donors[match_var] == recipient[match_var]
                    ]
            
            # If no exact matches, try partial matching (at least 50% match)
            if len(matching_donors) < donor_threshold:
                exact_match = False
                matching_donors = donors.copy()
                match_scores = []
                
                for donor_idx, donor in donors.iterrows():
                    score = 0
                    total_comparable = 0
                    
                    for match_var in match_vars:
                        if (match_var in recipient and match_var in donor and
                            pd.notna(recipient[match_var]) and pd.notna(donor[match_var])):
                            total_comparable += 1
                            if recipient[match_var] == donor[match_var]:
                                score += 1
                    
                    if total_comparable > 0:
                        match_ratio = score / total_comparable
                        match_scores.append((donor_idx, match_ratio))
                
                if match_scores:
                    # Get donors with at least 50% match
                    good_donors = [(idx, score) for idx, score in match_scores if score >= 0.5]
                    if good_donors:
                        # Get best matching donors
                        best_score = max(good_donors, key=lambda x: x[1])[1]
                        top_donors_idx = [idx for idx, score in good_donors if score == best_score]
                        matching_donors = donors.loc[top_donors_idx]
            
            # Select donor and impute value
            if len(matching_donors) >= donor_threshold:
                # Random selection from matching donors
                donor_idx = np.random.choice(matching_donors.index)
                donor = matching_donors.loc[donor_idx]
                imputed_value = donor[target_col]
                
                # Track donor usage
                donor_code = donor['Country Code']
                donor_usage[donor_code] = donor_usage.get(donor_code, 0) + 1
                
                # Determine match type
                match_type = 'exact' if exact_match else 'partial'
                
                log_entry = {
                    'recipient': recipient_name,
                    'recipient_code': recipient_code,
                    'donor': donor['Country'],
                    'donor_code': donor_code,
                    'imputed_value': float(imputed_value),
                    'n_matching_donors': len(matching_donors),
                    'match_type': match_type,
                    'match_score': best_score if not exact_match else 1.0
                }
                
                if exact_match:
                    fallback_counts['exact'] += 1
                else:
                    fallback_counts['partial'] += 1
                    
            else:
                # Fallback strategies
                # 1. Try mean of donors with same country band (if exists)
                if 'country_band' in match_vars and 'country_band' in recipient and pd.notna(recipient['country_band']):
                    band = recipient['country_band']
                    band_donors = donors[donors['country_band'] == band]
                    if len(band_donors) > 0:
                        imputed_value = band_donors[target_col].mean()
                        fallback_method = 'band_mean'
                        fallback_counts['band_mean'] += 1
                    else:
                        imputed_value = donors[target_col].mean()
                        fallback_method = 'global_mean'
                        fallback_counts['global_mean'] += 1
                else:
                    imputed_value = donors[target_col].mean()
                    fallback_method = 'global_mean'
                    fallback_counts['global_mean'] += 1
                
                log_entry = {
                    'recipient': recipient_name,
                    'recipient_code': recipient_code,
                    'donor': fallback_method,
                    'donor_code': fallback_method.upper(),
                    'imputed_value': float(imputed_value),
                    'n_matching_donsors': 0,
                    'match_type': 'fallback',
                    'fallback_method': fallback_method
                }
            
            # Update imputed value
            if pd.notna(imputed_value):
                imputed_data.loc[imputed_data['Country Code'] == recipient_code, target_col] = imputed_value
            
            col_log.append(log_entry)
        
        # Store log for this column
        imputation_log[target_col] = {
            'log': col_log,
            'donor_usage': donor_usage,
            'fallback_counts': fallback_counts,
            'n_imputed': len(col_log),
            'success_rate': sum(1 for entry in col_log if pd.notna(entry['imputed_value'])) / max(len(col_log), 1),
            'avg_match_donors': np.mean([entry['n_matching_donors'] for entry in col_log]) if col_log else 0
        }
        
        if verbose:
            print(f"Imputed: {len(col_log)} values")
            print(f"Unique donors used: {len(donor_usage)}")
            print(f"Success rate: {imputation_log[target_col]['success_rate']:.1%}")
            print(f"Fallback methods: {fallback_counts}")
    
    # Final summary
    if verbose:
        print(f"\n{'='*60}")
        print("HOT-DECK IMPUTATION COMPLETE")
        print(f"{'='*60}")
        
        total_imputed = 0
        for target_col in target_columns:
            if target_col in imputation_log:
                n_imputed = imputation_log[target_col]['n_imputed']
                success_rate = imputation_log[target_col]['success_rate']
                total_imputed += n_imputed
                print(f"{target_col}: {n_imputed} imputed ({success_rate:.1%} success)")
        
        print(f"\nTotal values imputed: {total_imputed}")
        print(f"Imputed data shape: {imputed_data.shape}")
        
        # Check remaining missing values
        remaining_missing = imputed_data[target_columns].isna().sum().sum()
        print(f"Remaining missing values: {remaining_missing}")
    
    return imputed_data, imputation_log


def evaluate_imputation(original_data, imputed_data, target_columns=None, 
                       mask=None, verbose=True):
    """
    Evaluate imputation accuracy against original data.
    
    Parameters:
    -----------
    original_data : DataFrame
        Original complete dataset (ground truth)
    imputed_data : DataFrame
        Dataset with imputed values
    target_columns : list, optional
        List of columns to evaluate. If None, evaluates all numeric columns
        except 'Country' and 'Country Code'.
    mask : array-like or DataFrame, optional
        Boolean mask indicating which values were imputed.
        If None, assumes all positions where imputed_data has values.
    verbose : bool
        Whether to print evaluation results
        
    Returns:
    --------
    dict with evaluation metrics per column
    """
    
    # Identify target columns
    if target_columns is None:
        numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [col for col in numeric_cols if col not in ['Country', 'Country Code']]
    
    # Ensure same countries are compared
    common_codes = set(original_data['Country Code']).intersection(
        set(imputed_data['Country Code'])
    )
    
    original_subset = original_data[original_data['Country Code'].isin(common_codes)].copy()
    imputed_subset = imputed_data[imputed_data['Country Code'].isin(common_codes)].copy()
    
    # Sort by Country Code for alignment
    original_subset = original_subset.sort_values('Country Code').reset_index(drop=True)
    imputed_subset = imputed_subset.sort_values('Country Code').reset_index(drop=True)
    
    metrics_by_column = {}
    
    for target_col in target_columns:
        if target_col not in original_subset.columns or target_col not in imputed_subset.columns:
            if verbose:
                print(f"Warning: {target_col} not found in both datasets, skipping")
            continue
        
        # If mask is provided, use it
        if mask is not None:
            if isinstance(mask, pd.DataFrame):
                col_mask = mask[target_col] if target_col in mask.columns else pd.Series(False, index=original_subset.index)
            else:
                col_mask = mask
        else:
            # Assume all positions with values in both datasets
            col_mask = original_subset[target_col].notna() & imputed_subset[target_col].notna()
        
        # Extract values for comparison
        true_values = original_subset.loc[col_mask, target_col]
        imputed_values = imputed_subset.loc[col_mask, target_col]
        
        # Align indices
        common_idx = true_values.index.intersection(imputed_values.index)
        true_values = true_values.loc[common_idx]
        imputed_values = imputed_values.loc[common_idx]
        
        if len(true_values) == 0:
            if verbose:
                print(f"Warning: No overlapping values for {target_col}")
            continue
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(true_values, imputed_values)
        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
        r2 = r2_score(true_values, imputed_values)
        correlation = true_values.corr(imputed_values)
        bias = np.mean(imputed_values - true_values)
        percent_error = np.mean(np.abs((imputed_values - true_values) / true_values)) * 100
        
        metrics_by_column[target_col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Correlation': correlation,
            'Bias': bias,
            'Percent_Error': percent_error,
            'n_evaluated': len(true_values),
            'avg_true_value': true_values.mean(),
            'avg_imputed_value': imputed_values.mean()
        }
        
        if verbose:
            print(f"\n{target_col}:")
            print(f"  Values evaluated: {len(true_values)}")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  R²: {r2:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            print(f"  Bias: {bias:.6f}")
            print(f"  Percent Error: {percent_error:.2f}%")
    
    if verbose and metrics_by_column:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY (averaged across columns)")
        print(f"{'='*60}")
        
        overall_metrics = {}
        for metric in ['MAE', 'RMSE', 'R2', 'Correlation', 'Bias', 'Percent_Error']:
            values = [metrics[metric] for metrics in metrics_by_column.values()]
            overall_metrics[metric] = np.mean(values)
            print(f"{metric}: {overall_metrics[metric]:.6f}")
    
    return metrics_by_column










def calculate_correlation_weights(original_df, covariate_df, target_columns, match_vars):
    """
    Calculate correlation between each target column and each covariate.
    Returns weight dictionary: {target_col: {covariate: weight}}
    """
    weights = {}
    
    # Merge target and covariate data
    merged = pd.merge(
        original_df[['Country Code'] + target_columns],
        covariate_df,
        on='Country Code',
        how='inner'
    )
    
    for target_col in target_columns:
        col_weights = {}
        
        for covar in match_vars:
            if covar in merged.columns and target_col in merged.columns:
                # Calculate absolute correlation
                valid_data = merged[[target_col, covar]].dropna()
                if len(valid_data) > 10:
                    corr = abs(valid_data[target_col].corr(valid_data[covar]))
                    
                    # Apply thresholds from paper
                    if corr >= 0.25:
                        col_weights[covar] = corr
                    elif corr >= 0.20:
                        col_weights[covar] = corr * 0.5  # Downweight
                    else:
                        col_weights[covar] = 0  # Exclude
                else:
                    col_weights[covar] = 0
        
        # Normalize weights to sum to 1
        total = sum(col_weights.values())
        if total > 0:
            col_weights = {k: v/total for k, v in col_weights.items()}
        
        weights[target_col] = col_weights
    
    return weights


def hotdeck_imputation_weighted(target_df, bands_df, original_df=None,
                               target_columns=None, match_vars=None, 
                               random_state=42, donor_threshold=1, 
                               weight_threshold=0.20, verbose=True):
    """
    Perform hot-deck imputation with correlation-based weights.
    
    Parameters:
    -----------
    target_df : DataFrame with missing values
    bands_df : DataFrame from create_country_bands_df()
    original_df : DataFrame, optional
        Original complete data for calculating correlation weights.
        If None, falls back to unweighted matching.
    target_columns : list of columns to impute
    match_vars : list of covariates to use for matching
    random_state : int
    donor_threshold : int, minimum matching donors required
    weight_threshold : float, minimum correlation to include covariate
    verbose : bool
    """
    
    # ============ VALIDATION ============
    required_target_cols = ['Country', 'Country Code']
    for col in required_target_cols:
        if col not in target_df.columns:
            raise ValueError(f"Target DataFrame missing '{col}' column")
    
    if 'Country Code' not in bands_df.columns:
        raise ValueError(f"Bands DataFrame missing 'Country Code' column")
    
    # ============ IDENTIFY TARGET COLUMNS ============
    if target_columns is None:
        numeric_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [col for col in numeric_cols if col not in ['Country', 'Country Code']]
    
    if verbose:
        print(f"Target columns to impute: {len(target_columns)}")
        print(f"Number of countries in target data: {len(target_df)}")
        print(f"Number of countries in bands data: {len(bands_df)}")
    
    # ============ MERGE TARGET WITH BANDS ============
    merged_data = pd.merge(
        target_df[['Country', 'Country Code'] + target_columns],
        bands_df,
        on='Country Code',
        how='left'
    )
    
    if verbose:
        print(f"Merged data shape: {merged_data.shape}")
        missing_total = merged_data[target_columns].isna().sum().sum()
        print(f"Missing values in merged data: {missing_total}")
    
    # ============ IDENTIFY MATCHING VARIABLES ============
    if match_vars is None:
        match_vars = []
        for col in bands_df.columns:
            if col not in ['Country Name', 'Country Code']:
                if pd.api.types.is_integer_dtype(bands_df[col]):
                    match_vars.append(col)
                elif (pd.api.types.is_float_dtype(bands_df[col]) and 
                      bands_df[col].dropna().isin([1.0, 2.0, 3.0, 4.0]).all()):
                    match_vars.append(col)
    
    if verbose:
        print(f"\nUsing {len(match_vars)} banded covariates for matching")
    
    # ============ CALCULATE CORRELATION WEIGHTS ============
    weights = None
    if original_df is not None:
        if verbose:
            print("\nCalculating correlation weights from original data...")
        
        # Merge original data with bands for correlation
        merged_original = pd.merge(
            original_df[['Country Code'] + target_columns],
            bands_df,
            on='Country Code',
            how='inner'
        )
        
        weights = {}
        for target_col in target_columns:
            col_weights = {}
            
            for covar in match_vars:
                if covar in merged_original.columns and target_col in merged_original.columns:
                    valid_data = merged_original[[target_col, covar]].dropna()
                    if len(valid_data) > 10:
                        corr = abs(valid_data[target_col].corr(valid_data[covar]))
                        
                        # Apply threshold
                        if corr >= weight_threshold:
                            col_weights[covar] = corr
                        else:
                            col_weights[covar] = 0
                    else:
                        col_weights[covar] = 0
            
            # Normalize weights to sum to 1
            total = sum(col_weights.values())
            if total > 0:
                col_weights = {k: v/total for k, v in col_weights.items()}
            
            weights[target_col] = col_weights
            
            if verbose:
                # Show top 3 predictors for first 5 columns
                if target_columns.index(target_col) < 5:
                    top_vars = sorted([(k, v) for k, v in col_weights.items() if v > 0], 
                                    key=lambda x: x[1], reverse=True)[:3]
                    if top_vars:
                        print(f"\n  {target_col} top predictors:")
                        for var, w in top_vars:
                            print(f"    {var}: weight={w:.3f}")
    
    # ============ INITIALIZE RESULTS ============
    imputed_data = target_df.copy()
    imputation_log = {}
    np.random.seed(random_state)
    
    # ============ PROCESS EACH TARGET COLUMN ============
    for target_col in target_columns:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing: {target_col}")
            print(f"{'='*50}")
        
        # Get weights for this column (or None if no weights)
        col_weights = weights[target_col] if weights is not None else None
        
        # Identify missing values
        missing_mask = merged_data[target_col].isna()
        if missing_mask.sum() == 0:
            if verbose:
                print(f"No missing values in {target_col}")
            continue
        
        recipients = merged_data[missing_mask].copy()
        donors = merged_data[~missing_mask].copy()
        
        if verbose:
            print(f"Missing values: {len(recipients)}")
            print(f"Available donors: {len(donors)}")
        
        col_log = []
        donor_usage = {}
        fallback_counts = {'exact': 0, 'weighted': 0, 'band_mean': 0, 'global_mean': 0}
        
        # Process each recipient
        for idx, recipient in recipients.iterrows():
            recipient_code = recipient['Country Code']
            recipient_name = recipient['Country']
            
            # ======== WEIGHTED MATCHING ========
            if col_weights is not None:
                # Calculate weighted match scores
                match_scores = []
                
                for donor_idx, donor in donors.iterrows():
                    score = 0
                    total_weight = 0
                    
                    for var in match_vars:
                        if var in col_weights and col_weights[var] > 0:
                            weight = col_weights[var]
                            total_weight += weight
                            
                            if (var in recipient and var in donor and
                                pd.notna(recipient[var]) and pd.notna(donor[var]) and
                                recipient[var] == donor[var]):
                                score += weight
                    
                    if total_weight > 0:
                        match_ratio = score / total_weight
                        match_scores.append((donor_idx, match_ratio))
                
                # Select best matches
                if match_scores:
                    match_scores.sort(key=lambda x: x[1], reverse=True)
                    best_score = match_scores[0][1]
                    
                    # Use donors with at least 80% of best score
                    threshold = best_score * 0.8
                    top_donors = [idx for idx, s in match_scores if s >= threshold]
                    
                    if len(top_donors) >= donor_threshold:
                        donor_idx = np.random.choice(top_donors)
                        donor = donors.loc[donor_idx]
                        imputed_value = donor[target_col]
                        
                        # Track donor usage
                        donor_code = donor['Country Code']
                        donor_usage[donor_code] = donor_usage.get(donor_code, 0) + 1
                        
                        log_entry = {
                            'recipient': recipient_name,
                            'recipient_code': recipient_code,
                            'donor': donor['Country'],
                            'donor_code': donor_code,
                            'imputed_value': float(imputed_value),
                            'n_matching_donors': len(top_donors),
                            'match_type': 'weighted',
                            'match_score': best_score
                        }
                        
                        fallback_counts['weighted'] += 1
                        imputed_data.loc[imputed_data['Country Code'] == recipient_code, target_col] = imputed_value
                        col_log.append(log_entry)
                        continue
            
            # ======== FALLBACK 1: EXACT MATCHING (if weighting failed or not available) ========
            matching_donors = donors.copy()
            for match_var in match_vars:
                if match_var in recipient and pd.notna(recipient[match_var]):
                    matching_donors = matching_donors[
                        matching_donors[match_var] == recipient[match_var]
                    ]
            
            if len(matching_donors) >= donor_threshold:
                donor_idx = np.random.choice(matching_donors.index)
                donor = matching_donors.loc[donor_idx]
                imputed_value = donor[target_col]
                
                donor_code = donor['Country Code']
                donor_usage[donor_code] = donor_usage.get(donor_code, 0) + 1
                
                log_entry = {
                    'recipient': recipient_name,
                    'recipient_code': recipient_code,
                    'donor': donor['Country'],
                    'donor_code': donor_code,
                    'imputed_value': float(imputed_value),
                    'n_matching_donors': len(matching_donors),
                    'match_type': 'exact'
                }
                
                fallback_counts['exact'] += 1
                imputed_data.loc[imputed_data['Country Code'] == recipient_code, target_col] = imputed_value
                col_log.append(log_entry)
                continue
            
            # ======== FALLBACK 2: BAND MEAN ========
            if 'country_band' in match_vars and 'country_band' in recipient and pd.notna(recipient['country_band']):
                band = recipient['country_band']
                band_donors = donors[donors['country_band'] == band]
                if len(band_donors) > 0:
                    imputed_value = band_donors[target_col].mean()
                    fallback_method = 'band_mean'
                    fallback_counts['band_mean'] += 1
                    
                    log_entry = {
                        'recipient': recipient_name,
                        'recipient_code': recipient_code,
                        'donor': f'band_mean_{band}',
                        'donor_code': 'BAND_MEAN',
                        'imputed_value': float(imputed_value),
                        'match_type': 'fallback',
                        'fallback_method': fallback_method
                    }
                    
                    imputed_data.loc[imputed_data['Country Code'] == recipient_code, target_col] = imputed_value
                    col_log.append(log_entry)
                    continue
            
            # ======== FALLBACK 3: GLOBAL MEAN ========
            imputed_value = donors[target_col].mean()
            fallback_counts['global_mean'] += 1
            
            log_entry = {
                'recipient': recipient_name,
                'recipient_code': recipient_code,
                'donor': 'global_mean',
                'donor_code': 'GLOBAL_MEAN',
                'imputed_value': float(imputed_value),
                'match_type': 'fallback',
                'fallback_method': 'global_mean'
            }
            
            imputed_data.loc[imputed_data['Country Code'] == recipient_code, target_col] = imputed_value
            col_log.append(log_entry)
        
        # Store log for this column
        imputation_log[target_col] = {
            'log': col_log,
            'donor_usage': donor_usage,
            'fallback_counts': fallback_counts,
            'n_imputed': len(col_log),
            'success_rate': len(col_log) / missing_mask.sum() if missing_mask.sum() > 0 else 0,
            'weighted_available': col_weights is not None
        }
        
        if verbose:
            print(f"Imputed: {len(col_log)} values")
            print(f"Unique donors used: {len(donor_usage)}")
            print(f"Fallback methods: {fallback_counts}")
    
    return imputed_data, imputation_log