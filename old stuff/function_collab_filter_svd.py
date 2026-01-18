import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SVDImputation:
    """
    SVD-based collaborative filtering for carbon emission data imputation
    Supports both country-wise and industry-wise approaches
    """
    
    def __init__(self, n_factors=5, scale_data=True):
        """
        Initialize SVD imputation model
        
        Parameters:
        -----------
        n_factors : int
            Number of singular values to keep (latent factors)
        scale_data : bool
            Whether to standardize the data (recommended for SVD)
        """
        self.n_factors = n_factors
        self.scale_data = scale_data
        self.scaler = StandardScaler()
        self.results = {}
        
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def svd_imputation_single(self, data, row_idx, col_idx, approach='country'):
        """
        Perform SVD imputation for a single missing value
        
        Parameters:
        -----------
        data : numpy array
            Complete data matrix
        row_idx : int
            Row index of missing value (country index)
        col_idx : int
            Column index of missing value (industry index)
        approach : str
            'country' or 'industry' - determines matrix orientation
        """
        # Create a copy of data with the target value masked
        data_masked = data.copy()
        data_masked[row_idx, col_idx] = np.nan
        
        if approach == 'industry':
            # Transpose for industry-wise approach
            data_masked = data_masked.T
        
        n_rows, n_cols = data_masked.shape
        
        # Fill missing values with column means
        data_filled = data_masked.copy()
        col_means = np.nanmean(data_filled, axis=0)
        
        for j in range(n_cols):
            mask_nan = np.isnan(data_filled[:, j])
            if np.any(mask_nan):
                data_filled[mask_nan, j] = col_means[j]
        
        # Perform truncated SVD
        # Using scipy's svds for efficiency (computes only k largest singular values)
        U, sigma, Vt = svds(data_filled, k=min(self.n_factors, min(n_rows, n_cols)-1))
        
        # Ensure singular values are in descending order
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        
        # Reconstruct matrix
        sigma_matrix = np.diag(sigma)
        reconstructed = U @ sigma_matrix @ Vt
        
        if approach == 'industry':
            reconstructed = reconstructed.T
            pred_row, pred_col = col_idx, row_idx
        else:
            pred_row, pred_col = row_idx, col_idx
        
        return reconstructed[pred_row, pred_col]
    
    def leave_one_out_svd(self, df, approach='country'):
        """
        Perform leave-one-out cross validation using SVD
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe with 'Country' column and industry columns
        approach : str
            'country' or 'industry'
        """
        # Extract data
        countries = df['Country'].values
        data = df.drop('Country', axis=1).values
        industries = df.drop('Country', axis=1).columns.tolist()
        
        # Scale data if requested
        if self.scale_data:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = data.copy()
        
        n_countries, n_industries = data_scaled.shape
        
        print(f"Starting leave-one-out SVD with {approach}-wise approach...")
        print(f"Data shape: {n_countries} countries × {n_industries} industries")
        print(f"Using {self.n_factors} latent factors")
        
        # Initialize arrays for results
        all_predictions = []
        all_actuals = []
        country_indices = []
        industry_indices = []
        
        # Get total number of iterations for progress bar
        total_iterations = n_countries * n_industries
        
        # Use tqdm for progress bar
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            for i in range(n_countries):
                for j in range(n_industries):
                    # Get actual value
                    actual_value = data_scaled[i, j]
                    
                    # Predict using SVD
                    predicted_value = self.svd_imputation_single(
                        data_scaled, i, j, approach
                    )
                    
                    # Store results
                    all_predictions.append(predicted_value)
                    all_actuals.append(actual_value)
                    country_indices.append(i)
                    industry_indices.append(j)
                    
                    pbar.update(1)
        
        # Convert predictions back to original scale if data was scaled
        if self.scale_data:
            # Reshape predictions to match data shape for inverse transform
            pred_matrix = np.zeros((n_countries, n_industries))
            act_matrix = np.zeros((n_countries, n_industries))
            
            for idx, (c_idx, i_idx) in enumerate(zip(country_indices, industry_indices)):
                pred_matrix[c_idx, i_idx] = all_predictions[idx]
                act_matrix[c_idx, i_idx] = all_actuals[idx]
            
            # Inverse transform
            pred_original = self.scaler.inverse_transform(pred_matrix)
            act_original = self.scaler.inverse_transform(act_matrix)
            
            # Flatten back
            all_predictions_original = []
            all_actuals_original = []
            
            for idx, (c_idx, i_idx) in enumerate(zip(country_indices, industry_indices)):
                all_predictions_original.append(pred_original[c_idx, i_idx])
                all_actuals_original.append(act_original[c_idx, i_idx])
            
            all_predictions = all_predictions_original
            all_actuals = all_actuals_original
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        mae = mean_absolute_error(all_actuals, all_predictions)
        mape = self.calculate_mape(all_actuals, all_predictions)
        
        # Store detailed results
        results = {
            'approach': approach,
            'n_factors': self.n_factors,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals),
            'country_indices': np.array(country_indices),
            'industry_indices': np.array(industry_indices),
            'countries': countries,
            'industries': industries,
            'data_shape': (n_countries, n_industries)
        }
        
        self.results[approach] = results
        
        return results
    
    def evaluate_both_approaches(self, df):
        """
        Evaluate both country-wise and industry-wise SVD approaches
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        """
        print("=" * 70)
        print("SVD COLLABORATIVE FILTERING EVALUATION")
        print("=" * 70)
        
        approaches = ['country', 'industry']
        all_results = {}
        
        for approach in approaches:
            print(f"\n{'='*50}")
            print(f"Evaluating {approach.upper()}-wise approach")
            print('='*50)
            
            results = self.leave_one_out_svd(df, approach=approach)
            all_results[approach] = results
            
            print(f"\nResults for {approach}-wise SVD:")
            print(f"  RMSE:  {results['rmse']:.6f}")
            print(f"  R²:    {results['r2']:.6f}")
            print(f"  MAE:   {results['mae']:.6f}")
            print(f"  MAPE:  {results['mape']:.6f}%")
        
        return all_results
    
    def analyze_country_performance(self, results, top_n=10):
        """
        Analyze performance by country
        
        Parameters:
        -----------
        results : dict
            Results from leave_one_out_svd
        top_n : int
            Number of top/bottom countries to show
        """
        approach = results['approach']
        countries = results['countries']
        country_indices = results['country_indices']
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Calculate RMSE per country
        country_rmse = {}
        country_counts = {}
        
        for idx, country_idx in enumerate(country_indices):
            country_name = countries[country_idx]
            error = (actuals[idx] - predictions[idx]) ** 2
            
            if country_name not in country_rmse:
                country_rmse[country_name] = 0
                country_counts[country_name] = 0
            
            country_rmse[country_name] += error
            country_counts[country_name] += 1
        
        # Compute average RMSE per country
        country_avg_rmse = {}
        for country in country_rmse:
            country_avg_rmse[country] = np.sqrt(country_rmse[country] / country_counts[country])
        
        # Sort countries by RMSE
        sorted_countries = sorted(country_avg_rmse.items(), key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"COUNTRY-WISE PERFORMANCE ANALYSIS ({approach.upper()}-wise approach)")
        print('='*70)
        
        print(f"\nTop {top_n} countries with LOWEST RMSE (best predictions):")
        for i, (country, rmse) in enumerate(sorted_countries[:top_n]):
            print(f"  {i+1:2d}. {country:30s} RMSE: {rmse:.6f}")
        
        print(f"\nTop {top_n} countries with HIGHEST RMSE (worst predictions):")
        for i, (country, rmse) in enumerate(sorted_countries[-top_n:][::-1]):
            print(f"  {i+1:2d}. {country:30s} RMSE: {rmse:.6f}")
        
        return country_avg_rmse
    
    def analyze_industry_performance(self, results, top_n=10):
        """
        Analyze performance by industry
        
        Parameters:
        -----------
        results : dict
            Results from leave_one_out_svd
        top_n : int
            Number of top/bottom industries to show
        """
        approach = results['approach']
        industries = results['industries']
        industry_indices = results['industry_indices']
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Calculate RMSE per industry
        industry_rmse = {}
        industry_counts = {}
        
        for idx, industry_idx in enumerate(industry_indices):
            industry_name = industries[industry_idx]
            error = (actuals[idx] - predictions[idx]) ** 2
            
            if industry_name not in industry_rmse:
                industry_rmse[industry_name] = 0
                industry_counts[industry_name] = 0
            
            industry_rmse[industry_name] += error
            industry_counts[industry_name] += 1
        
        # Compute average RMSE per industry
        industry_avg_rmse = {}
        for industry in industry_rmse:
            industry_avg_rmse[industry] = np.sqrt(industry_rmse[industry] / industry_counts[industry])
        
        # Sort industries by RMSE
        sorted_industries = sorted(industry_avg_rmse.items(), key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"INDUSTRY-WISE PERFORMANCE ANALYSIS ({approach.upper()}-wise approach)")
        print('='*70)
        
        print(f"\nTop {top_n} industries with LOWEST RMSE (best predictions):")
        for i, (industry, rmse) in enumerate(sorted_industries[:top_n]):
            print(f"  {i+1:2d}. {industry:20s} RMSE: {rmse:.6f}")
        
        print(f"\nTop {top_n} industries with HIGHEST RMSE (worst predictions):")
        for i, (industry, rmse) in enumerate(sorted_industries[-top_n:][::-1]):
            print(f"  {i+1:2d}. {industry:20s} RMSE: {rmse:.6f}")
        
        return industry_avg_rmse
    
    def get_summary_dataframe(self):
        """Create a summary DataFrame of all results"""
        if not self.results:
            print("No results available. Run evaluate_both_approaches() first.")
            return None
        
        summary_data = []
        for approach, result in self.results.items():
            summary_data.append({
                'Approach': f"SVD ({approach}-wise)",
                'Latent_Factors': result['n_factors'],
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'MAPE (%)': result['mape']
            })
        
        return pd.DataFrame(summary_data)
    
    def visualize_singular_values(self, df):
        """
        Visualize singular values to help choose optimal n_factors
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        """
        import matplotlib.pyplot as plt
        
        # Extract and scale data
        data = df.drop('Country', axis=1).values
        if self.scale_data:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = data
        
        # Compute full SVD (for visualization only - might be slow for large matrices)
        print("\nComputing singular values for visualization...")
        U, s, Vt = np.linalg.svd(data_scaled, full_matrices=False)
        
        # Plot singular values
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(s) + 1), s, 'bo-', linewidth=2)
        plt.axvline(x=self.n_factors, color='r', linestyle='--', 
                   label=f'Current n_factors = {self.n_factors}')
        plt.xlabel('Singular Value Index', fontsize=12)
        plt.ylabel('Singular Value Magnitude', fontsize=12)
        plt.title('Singular Values of Carbon Emissions Data', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot cumulative variance explained
        variance_explained = (s ** 2) / np.sum(s ** 2)
        cumulative_variance = np.cumsum(variance_explained)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', linewidth=2)
        plt.axhline(y=0.8, color='g', linestyle='--', label='80% variance')
        plt.axhline(y=0.9, color='b', linestyle='--', label='90% variance')
        plt.axvline(x=self.n_factors, color='r', linestyle='--', 
                   label=f'Current n_factors = {self.n_factors}')
        plt.xlabel('Number of Singular Values', fontsize=12)
        plt.ylabel('Cumulative Variance Explained', fontsize=12)
        plt.title('Cumulative Variance Explained by Singular Values', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nVariance explained by first {self.n_factors} singular values: {cumulative_variance[self.n_factors-1]:.2%}")
        print(f"Number of singular values needed for 80% variance: {np.argmax(cumulative_variance >= 0.8) + 1}")
        print(f"Number of singular values needed for 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
        
        return s, cumulative_variance


# Convenience function for quick analysis
def run_svd_analysis(df, n_factors=5, scale_data=True):
    """
    Run complete SVD analysis pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Your carbon emissions dataframe
    n_factors : int
        Number of latent factors
    scale_data : bool
        Whether to scale data
        
    Returns:
    --------
    svd_model : SVDImputation object
    all_results : dict
        Results from both approaches
    """
    # Initialize model
    svd_model = SVDImputation(n_factors=n_factors, scale_data=scale_data)
    
    # Run evaluation
    all_results = svd_model.evaluate_both_approaches(df)
    
    # Get summary
    summary_df = svd_model.get_summary_dataframe()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Analyze country performance for both approaches
    for approach in ['country', 'industry']:
        if approach in all_results:
            print(f"\nAnalyzing {approach}-wise performance...")
            svd_model.analyze_country_performance(all_results[approach], top_n=5)
            svd_model.analyze_industry_performance(all_results[approach], top_n=5)
    
    return svd_model, all_results, summary_df