import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def baseline_imputation_leave_one_out(df, method='combined'):
    """
    Perform leave-one-out cross-validation for baseline imputation methods.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data with countries as rows and industries as columns
    method : str
        One of: 'country_mean', 'industry_mean', 'combined'
    
    Returns:
    --------
    tuple : (true_values, predicted_values, metrics_dict)
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Extract numeric data (excluding 'Country' column)
    countries = data['Country']
    numeric_data = data.drop('Country', axis=1).values
    n_countries, n_industries = numeric_data.shape
    
    # Store results
    true_values = []
    predicted_values = []
    
    # For each data point (country, industry)
    for i in range(n_countries):
        for j in range(n_industries):
            # Store true value
            true_val = numeric_data[i, j]
            true_values.append(true_val)
            
            if method == 'country_mean':
                # Calculate mean of country i excluding value at column j
                country_vals = np.delete(numeric_data[i, :], j)
                valid_vals = country_vals[~np.isnan(country_vals)]
                pred = np.mean(valid_vals) if len(valid_vals) > 0 else 0
            
            elif method == 'industry_mean':
                # Calculate mean of industry j excluding value at row i
                industry_vals = np.delete(numeric_data[:, j], i)
                valid_vals = industry_vals[~np.isnan(industry_vals)]
                pred = np.mean(valid_vals) if len(valid_vals) > 0 else 0
            
            elif method == 'combined':
                # Calculate both means and combine
                country_vals = np.delete(numeric_data[i, :], j)
                industry_vals = np.delete(numeric_data[:, j], i)
                
                country_valid = country_vals[~np.isnan(country_vals)]
                industry_valid = industry_vals[~np.isnan(industry_vals)]
                
                country_mean = np.mean(country_valid) if len(country_valid) > 0 else 0
                industry_mean = np.mean(industry_valid) if len(industry_valid) > 0 else 0
                pred = 0.5 * country_mean + 0.5 * industry_mean
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            predicted_values.append(pred)
    
    # Convert to numpy arrays
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    
    # Calculate MAPE safely (avoid division by zero)
    absolute_percentage_errors = []
    for true, pred in zip(true_values, predicted_values):
        if true != 0:  # Only calculate if true value is not zero
            absolute_percentage_errors.append(np.abs((true - pred) / true))
    
    mape = np.mean(absolute_percentage_errors) * 100 if absolute_percentage_errors else 0
    
    metrics = {
        'RMSE': rmse,
        'R2': r2,
        'MAE': np.mean(np.abs(true_values - predicted_values)),
        'Mean Absolute Percentage Error': mape
    }
    
    return true_values, predicted_values, metrics

def compare_baseline_methods(df):
    """
    Compare all three baseline methods and return results.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    
    Returns:
    --------
    dict : Results for each method
    """
    methods = ['country_mean', 'industry_mean', 'combined']
    results = {}
    
    print("=" * 60)
    print("BASELINE MODELS COMPARISON - LEAVE-ONE-OUT CROSS VALIDATION")
    print("=" * 60)
    print(f"Dataset shape: {df.shape[0]} countries × {df.shape[1]-1} industries")
    print(f"Total data points: {df.shape[0] * (df.shape[1]-1)}")
    print("=" * 60)
    
    for method in methods:
        print(f"\nMethod: {method.replace('_', ' ').title()}")
        print("-" * 40)
        
        true_vals, pred_vals, metrics = baseline_imputation_leave_one_out(df, method)
        
        results[method] = {
            'true_values': true_vals,
            'predicted_values': pred_vals,
            'metrics': metrics
        }
        
        # Print metrics
        for metric_name, value in metrics.items():
            if metric_name == 'R2':
                print(f"{metric_name}: {value:.6f}")
            elif metric_name == 'Mean Absolute Percentage Error':
                print(f"{metric_name}: {value:.2f}%")
            else:
                print(f"{metric_name}: {value:.6f}")
    
    return results

def create_summary_table(results):
    """
    Create a summary comparison table of all methods.
    
    Parameters:
    -----------
    results : dict
        Results from compare_baseline_methods
    
    Returns:
    --------
    pandas DataFrame : Summary table
    """
    summary_data = []
    
    for method, result in results.items():
        metrics = result['metrics']
        summary_data.append({
            'Method': method.replace('_', ' ').title(),
            'RMSE': metrics['RMSE'],
            'R²': metrics['R2'],
            'MAE': metrics['MAE'],
            'MAPE (%)': metrics['Mean Absolute Percentage Error']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('RMSE')  # Sort by best RMSE
    
    return summary_df

def visualize_results(results):
    """
    Create visualizations comparing the methods.
    
    Parameters:
    -----------
    results : dict
        Results from compare_baseline_methods
    """
    import matplotlib.pyplot as plt
    
    methods = list(results.keys())
    method_names = [m.replace('_', '\n').title() for m in methods]
    r2_values = [results[m]['metrics']['R2'] for m in methods]
    rmse_values = [results[m]['metrics']['RMSE'] for m in methods]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Bar chart for R²
    axes[0, 0].bar(range(len(methods)), r2_values)
    axes[0, 0].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels(method_names, rotation=0)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(r2_values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Bar chart for RMSE
    axes[0, 1].bar(range(len(methods)), rmse_values)
    axes[0, 1].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(method_names, rotation=0)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Bar chart for MAE
    mae_values = [results[m]['metrics']['MAE'] for m in methods]
    axes[0, 2].bar(range(len(methods)), mae_values)
    axes[0, 2].set_title('MAE Comparison', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(range(len(methods)))
    axes[0, 2].set_xticklabels(method_names, rotation=0)
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mae_values):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Scatter plots for each method
    for idx, method in enumerate(methods):
        row = 1
        col = idx
        
        true_vals = results[method]['true_values']
        pred_vals = results[method]['predicted_values']
        
        ax = axes[row, col]
        scatter = ax.scatter(true_vals, pred_vals, alpha=0.3, s=10, c='blue')
        
        # Add perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{method.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of errors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, method in enumerate(methods):
        true_vals = results[method]['true_values']
        pred_vals = results[method]['predicted_values']
        errors = pred_vals - true_vals
        
        axes[idx].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[idx].set_title(f'{method.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Prediction Error')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        stats_text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nBias: {mean_error:.4f}'
        axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes, 
                      verticalalignment='top', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def run_baseline_analysis(df):
    """
    Simplified version that runs all three baseline methods.
    """
    # Prepare data
    countries = df['Country']
    data = df.drop('Country', axis=1).values
    n_rows, n_cols = data.shape
    
    # Initialize storage
    methods = ['country_mean', 'industry_mean', 'combined']
    predictions = {m: [] for m in methods}
    truths = []
    
    # Leave-one-out cross-validation
    for i in range(n_rows):
        for j in range(n_cols):
            true_val = data[i, j]
            truths.append(true_val)
            
            # Country mean (excluding current column)
            country_vals = np.delete(data[i, :], j)
            valid_country = country_vals[~np.isnan(country_vals)]
            country_mean = np.mean(valid_country) if len(valid_country) > 0 else 0
            
            # Industry mean (excluding current row)
            industry_vals = np.delete(data[:, j], i)
            valid_industry = industry_vals[~np.isnan(industry_vals)]
            industry_mean = np.mean(valid_industry) if len(valid_industry) > 0 else 0
            
            # Store predictions for each method
            predictions['country_mean'].append(country_mean)
            predictions['industry_mean'].append(industry_mean)
            predictions['combined'].append(0.5 * country_mean + 0.5 * industry_mean)
    
    # Convert to arrays
    truths = np.array(truths)
    
    # Calculate metrics for each method
    results = {}
    for method in methods:
        preds = np.array(predictions[method])
        
        rmse = np.sqrt(mean_squared_error(truths, preds))
        r2 = r2_score(truths, preds)
        
        # Calculate MAPE safely
        absolute_percentage_errors = []
        for true, pred in zip(truths, preds):
            if true != 0:
                absolute_percentage_errors.append(np.abs((true - pred) / true))
        
        mape = np.mean(absolute_percentage_errors) * 100 if absolute_percentage_errors else 0
        
        results[method] = {
            'RMSE': rmse,
            'R2': r2,
            'MAE': np.mean(np.abs(truths - preds)),
            'MAPE': mape
        }
    
    return results

# Example usage with your data
if __name__ == "__main__":
    # Load your data (replace with your actual data loading)
    # df = pd.read_csv('your_data.csv')
    
    # For demonstration, let's create a sample dataset structure
    # Replace this with your actual dataframe loading
    
    print("Loading and preparing data...")
    # Your actual data loading here