import pandas as pd
import numpy as np

# 1. Load your data
masked_df = pd.read_csv('ceda_benchmark_10pct_seed42.csv')  # Your masked data
reference_df = pd.read_csv('masked_values_reference.csv')   # The answer key

# 2. Apply your imputation algorithm to the masked data
def your_imputation_algorithm(df):
    """Your imputation code goes here"""
    # Example: simple mean imputation
    imputed = df.copy()
    numeric_cols = imputed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        imputed[col] = imputed[col].fillna(imputed[col].mean())
    return imputed

imputed_df = your_imputation_algorithm(masked_df)

# 3. SIMPLE EVALUATION - Just merge and compare!
# Create a merged DataFrame for easy comparison
results = []

for _, ref_row in reference_df.iterrows():
    country = ref_row['Country']
    sector = ref_row['Sector']
    true_value = ref_row['Original_Value']
    
    # Get your imputed value directly
    imputed_value = imputed_df.loc[
        imputed_df['Country'] == country, sector
    ].values[0]
    
    results.append({
        'Country': country,
        'Sector': sector,
        'True_Value': true_value,
        'Imputed_Value': imputed_value,
        'Error': true_value - imputed_value,
        'Abs_Error': abs(true_value - imputed_value)
    })

results_df = pd.DataFrame(results)

# 4. Calculate metrics (super simple!)
mse = np.mean(results_df['Error'] ** 2)
rmse = np.sqrt(mse)
mae = np.mean(results_df['Abs_Error'])
r2 = 1 - (np.sum(results_df['Error'] ** 2) / 
          np.sum((results_df['True_Value'] - results_df['True_Value'].mean()) ** 2))

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.4f}")

# 5. Save the comparison
results_df.to_csv('imputation_results.csv', index=False)
print(f"\nSaved detailed results to 'imputation_results.csv'")