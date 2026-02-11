import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_mask_dict(masked_df, imputed_df, columns_with_missing):
    mask_dict = {}
    for col in columns_with_missing:
        if col in masked_df.columns and col in imputed_df.columns:
            mask_dict[col] = masked_df[col].isna() & imputed_df[col].notna()
    
    print(f"Columns with imputed values: {len([k for k, v in mask_dict.items() if v.any()])}")
    return mask_dict

def calculate_metrics_bulk(original_df, imputed_df, mask_dict):
    results = []
    
    for col, mask in mask_dict.items():
        if mask.sum() == 0:
            continue
            
        imputed_indices = imputed_df[mask].index
        
        true_vals = []
        imputed_vals = []
        country_codes = []
        
        for idx in imputed_indices:
            country_code = imputed_df.loc[idx, 'Country Code']
            
            true_match = original_df[original_df['Country Code'] == country_code]
            imputed_match = imputed_df[imputed_df['Country Code'] == country_code]
            
            if len(true_match) > 0 and len(imputed_match) > 0:
                true_val = true_match[col].values[0]
                imputed_val = imputed_match[col].values[0]
                
                if pd.notna(true_val) and pd.notna(imputed_val):
                    true_vals.append(true_val)
                    imputed_vals.append(imputed_val)
                    country_codes.append(country_code)
        
        if len(true_vals) > 1:
            true_vals = np.array(true_vals)
            imputed_vals = np.array(imputed_vals)
            
            mae = mean_absolute_error(true_vals, imputed_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, imputed_vals))
            
            try:
                correlation = np.corrcoef(true_vals, imputed_vals)[0, 1]
            except:
                correlation = np.nan
            
            ss_res = np.sum((true_vals - imputed_vals) ** 2)
            ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            results.append({
                'column': col,
                'n_imputed_total': mask.sum(),
                'n_evaluated': len(true_vals),
                'MAE': mae,
                'RMSE': rmse,
                'Correlation': correlation,
                'R2': r2,
                'Bias': np.mean(imputed_vals - true_vals),
                'Percent_Error': np.mean(np.abs((imputed_vals - true_vals) / true_vals)) * 100 if np.all(true_vals != 0) else np.nan,
                'avg_true': np.mean(true_vals),
                'avg_imputed': np.mean(imputed_vals)
            })
        elif len(true_vals) == 1:
            mae = np.abs(true_vals[0] - imputed_vals[0])
            rmse = mae
            
            results.append({
                'column': col,
                'n_imputed_total': mask.sum(),
                'n_evaluated': 1,
                'MAE': mae,
                'RMSE': rmse,
                'Correlation': np.nan,
                'R2': np.nan,
                'Bias': imputed_vals[0] - true_vals[0],
                'Percent_Error': np.abs((imputed_vals[0] - true_vals[0]) / true_vals[0]) * 100 if true_vals[0] != 0 else np.nan,
                'avg_true': true_vals[0],
                'avg_imputed': imputed_vals[0]
            })
    
    return pd.DataFrame(results)

def evaluate_imputation(original_df, imputed_df, masked_df, columns_with_missing, config=None):
    mask_dict = create_mask_dict(masked_df, imputed_df, columns_with_missing)
    
    print("Calculating evaluation metrics...")
    eval_df = calculate_metrics_bulk(original_df, imputed_df, mask_dict)
    
    if len(eval_df) > 0 and config and 'output_dir' in config:
        eval_path = os.path.join(config['output_dir'], 'evaluation_metrics_all_columns.xlsx')
        eval_df.to_excel(eval_path, index=False)
        print(f"✓ Evaluation metrics saved to: {eval_path}")
        
        print(f"\nEvaluation Summary:")
        print(f"  Columns evaluated: {len(eval_df)}")
        print(f"  Total imputed values across all columns: {eval_df['n_imputed_total'].sum()}")
        print(f"  Total successfully evaluated: {eval_df['n_evaluated'].sum()}")
        print(f"  Average MAE: {eval_df['MAE'].mean():.6f}")
        print(f"  Average RMSE: {eval_df['RMSE'].mean():.6f}")
        
        if len(eval_df) > 0:
            print(f"\nTop 5 best performing columns (lowest MAE):")
            best_cols = eval_df.nsmallest(5, 'MAE')[['column', 'MAE', 'n_evaluated']]
            for _, row in best_cols.iterrows():
                print(f"  {row['column']}: MAE={row['MAE']:.6f} (n={row['n_evaluated']})")
            
            print(f"\nTop 5 worst performing columns (highest MAE):")
            worst_cols = eval_df.nlargest(5, 'MAE')[['column', 'MAE', 'n_evaluated']]
            for _, row in worst_cols.iterrows():
                print(f"  {row['column']}: MAE={row['MAE']:.6f} (n={row['n_evaluated']})")
    else:
        print("⚠️ No columns could be evaluated - check if original data has values for imputed positions")
    
    print(f"\nColumns with remaining missing values:")
    remaining_missing = []
    for col in columns_with_missing:
        if col in imputed_df.columns:
            still_missing = imputed_df[col].isna().sum()
            if still_missing > 0:
                remaining_missing.append({
                    'column': col,
                    'still_missing': still_missing,
                    'originally_missing': masked_df[col].isna().sum() if col in masked_df.columns else 0
                })

    if remaining_missing:
        remaining_df = pd.DataFrame(remaining_missing)
        remaining_df = remaining_df.sort_values('still_missing', ascending=False)
        print(remaining_df.head(10).to_string(index=False))
        if len(remaining_df) > 10:
            print(f"  ... and {len(remaining_df) - 10} more columns")
        
        if config and 'output_dir' in config:
            remaining_path = os.path.join(config['output_dir'], 'columns_not_fully_imputed.xlsx')
            remaining_df.to_excel(remaining_path, index=False)
            print(f"✓ Columns with remaining missing values saved to: {remaining_path}")
    else:
        print("  ✅ All missing values were successfully imputed!")
    
    return eval_df