import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class KNNCollaborativeFiltering:
    """
    k-Nearest Neighbors with Cosine Similarity for carbon emission data imputation
    Implements leave-one-out cross validation with both country-wise and industry-wise approaches
    """
    
    def __init__(self, k_neighbors=5, scale_data=True):
        """
        Initialize k-NN collaborative filtering model
        
        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to consider
        scale_data : bool
            Whether to standardize the data (recommended for cosine similarity)
        """
        self.k_neighbors = k_neighbors
        self.scale_data = scale_data
        self.scaler = StandardScaler()
        self.results = {}
        
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        
        Parameters:
        -----------
        vec1, vec2 : numpy arrays
            Input vectors
            
        Returns:
        --------
        similarity : float
            Cosine similarity between vectors
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def compute_similarity_matrix(self, data, approach='country'):
        """
        Compute cosine similarity matrix
        
        Parameters:
        -----------
        data : numpy array
            Input data matrix
        approach : str
            'country' or 'industry'
            
        Returns:
        --------
        similarity_matrix : numpy array
            Cosine similarity matrix
        """
        if approach == 'industry':
            # For industry-wise, we work with transposed data
            data = data.T
        
        n_samples = data.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        print(f"Computing cosine similarity matrix for {approach}-wise approach...")
        
        # Compute similarities (upper triangle only for efficiency)
        for i in tqdm(range(n_samples), desc="Computing similarities"):
            for j in range(i, n_samples):
                if i == j:
                    similarity = 1.0  # Self-similarity
                else:
                    # Get vectors, handling any NaN values
                    vec1 = data[i]
                    vec2 = data[j]
                    
                    # Create mask for non-NaN values in both vectors
                    mask = ~np.isnan(vec1) & ~np.isnan(vec2)
                    
                    if np.sum(mask) > 0:  # Only compute if there are common non-NaN values
                        similarity = self.cosine_similarity(vec1[mask], vec2[mask])
                    else:
                        similarity = 0  # No common data points
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Set diagonal to 0 for neighbor selection (don't consider self as neighbor)
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def predict_missing_value(self, data, target_row, target_col, 
                             similarity_matrix, approach='country'):
        """
        Predict a single missing value using k-nearest neighbors
        
        Parameters:
        -----------
        data : numpy array
            Input data matrix
        target_row : int
            Row index of value to predict
        target_col : int
            Column index of value to predict
        similarity_matrix : numpy array
            Precomputed similarity matrix
        approach : str
            'country' or 'industry'
            
        Returns:
        --------
        predicted_value : float
            Predicted value
        neighbor_indices : list
            Indices of neighbors used
        neighbor_similarities : list
            Similarities of neighbors used
        """
        if approach == 'industry':
            # Adjust indices for transposed data
            target_row_adj = target_col
            target_col_adj = target_row
        else:
            target_row_adj = target_row
            target_col_adj = target_col
        
        # Get similarities for the target
        similarities = similarity_matrix[target_row_adj]
        
        # Get indices of k nearest neighbors (highest similarity)
        # Exclude negative similarities (dissimilar items)
        valid_indices = np.where(similarities > 0)[0]
        
        if len(valid_indices) == 0:
            # No similar neighbors found, use column mean
            column_data = data[:, target_col] if approach == 'country' else data[target_col, :]
            valid_vals = column_data[~np.isnan(column_data)]
            if len(valid_vals) > 0:
                return np.mean(valid_vals), [], []
            else:
                return 0, [], []
        
        # Sort valid indices by similarity (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Take top k neighbors
        k = min(self.k_neighbors, len(sorted_indices))
        neighbor_indices = sorted_indices[:k]
        
        # Collect predictions from neighbors
        predictions = []
        weights = []
        
        for neighbor_idx in neighbor_indices:
            if approach == 'country':
                neighbor_value = data[neighbor_idx, target_col]
            else:
                neighbor_value = data[target_row, neighbor_idx]  # Note: data is not transposed here
            
            # Only use neighbors that have a value for this column
            if not np.isnan(neighbor_value):
                weight = similarities[neighbor_idx]
                predictions.append(neighbor_value)
                weights.append(weight)
        
        if len(predictions) > 0:
            # Weighted average prediction
            predictions = np.array(predictions)
            weights = np.array(weights)
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            predicted_value = np.sum(predictions * weights)
            return predicted_value, neighbor_indices[:len(predictions)], weights
        else:
            # Fallback: column mean
            column_data = data[:, target_col] if approach == 'country' else data[target_row, :]
            valid_vals = column_data[~np.isnan(column_data)]
            if len(valid_vals) > 0:
                return np.mean(valid_vals), [], []
            else:
                return 0, [], []
    
    def leave_one_out_cross_validation(self, df, approach='country'):
        """
        Perform leave-one-out cross validation using k-NN
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe with 'Country' column and industry columns
        approach : str
            'country' or 'industry'
            
        Returns:
        --------
        results : dict
            Dictionary containing all results and metrics
        """
        # Extract data
        countries = df['Country'].values
        data_original = df.drop('Country', axis=1).values
        industries = df.drop('Country', axis=1).columns.tolist()
        
        # Scale data if requested
        if self.scale_data:
            data_scaled = self.scaler.fit_transform(data_original)
        else:
            data_scaled = data_original.copy()
        
        n_countries, n_industries = data_scaled.shape
        
        print(f"\nStarting leave-one-out k-NN with {approach}-wise approach")
        print(f"Data shape: {n_countries} countries × {n_industries} industries")
        print(f"Number of neighbors (k): {self.k_neighbors}")
        print(f"Total data points: {n_countries * n_industries:,}")
        
        # Precompute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(data_scaled, approach)
        
        # Initialize arrays for results
        all_predictions = []
        all_actuals = []
        all_neighbors_used = []
        country_indices = []
        industry_indices = []
        
        total_iterations = n_countries * n_industries
        
        # Perform leave-one-out validation
        with tqdm(total=total_iterations, desc="Leave-one-out CV") as pbar:
            for i in range(n_countries):
                for j in range(n_industries):
                    # Store indices
                    country_indices.append(i)
                    industry_indices.append(j)
                    
                    # Get actual value
                    actual_value = data_scaled[i, j]
                    all_actuals.append(actual_value)
                    
                    # Predict using k-NN
                    predicted_value, neighbors, weights = self.predict_missing_value(
                        data_scaled, i, j, similarity_matrix, approach
                    )
                    
                    all_predictions.append(predicted_value)
                    all_neighbors_used.append(len(neighbors))
                    
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
            all_predictions_final = []
            all_actuals_final = []
            
            for idx, (c_idx, i_idx) in enumerate(zip(country_indices, industry_indices)):
                all_predictions_final.append(pred_original[c_idx, i_idx])
                all_actuals_final.append(act_original[c_idx, i_idx])
            
            all_predictions = all_predictions_final
            all_actuals = all_actuals_final
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        mae = mean_absolute_error(all_actuals, all_predictions)
        mape = self.calculate_mape(all_actuals, all_predictions)
        
        # Calculate average neighbors used
        avg_neighbors_used = np.mean(all_neighbors_used)
        
        # Store detailed results
        results = {
            'approach': approach,
            'k_neighbors': self.k_neighbors,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'avg_neighbors_used': avg_neighbors_used,
            'predictions': np.array(all_predictions),
            'actuals': np.array(all_actuals),
            'country_indices': np.array(country_indices),
            'industry_indices': np.array(industry_indices),
            'neighbors_used': np.array(all_neighbors_used),
            'countries': countries,
            'industries': industries,
            'similarity_matrix': similarity_matrix,
            'data_shape': (n_countries, n_industries),
            'scaler': self.scaler if self.scale_data else None
        }
        
        self.results[approach] = results
        
        return results
    
    def evaluate_both_approaches(self, df):
        """
        Evaluate both country-wise and industry-wise approaches
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
            
        Returns:
        --------
        all_results : dict
            Dictionary with results for both approaches
        """
        print("=" * 70)
        print("k-NN COLLABORATIVE FILTERING - BOTH APPROACHES")
        print("=" * 70)
        
        approaches = ['country', 'industry']
        all_results = {}
        
        for approach in approaches:
            print(f"\n{'='*50}")
            print(f"Evaluating {approach.upper()}-wise k-NN")
            print('='*50)
            
            results = self.leave_one_out_cross_validation(df, approach=approach)
            all_results[approach] = results
            
            print(f"\nPerformance metrics for {approach}-wise k-NN:")
            print(f"  RMSE:                  {results['rmse']:.6f}")
            print(f"  R² Score:              {results['r2']:.6f}")
            print(f"  MAE:                   {results['mae']:.6f}")
            print(f"  MAPE:                  {results['mape']:.6f}%")
            print(f"  Average neighbors used: {results['avg_neighbors_used']:.2f}")
        
        return all_results
    
    def get_summary_dataframe(self):
        """
        Create summary DataFrame of results
        
        Returns:
        --------
        summary_df : pandas DataFrame
            Summary of results for both approaches
        """
        if not self.results:
            print("No results available. Run evaluate_both_approaches() first.")
            return None
        
        summary_data = []
        for approach, result in self.results.items():
            summary_data.append({
                'Approach': f"k-NN ({approach}-wise)",
                'k': result['k_neighbors'],
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'MAPE (%)': result['mape'],
                'Avg Neighbors Used': result['avg_neighbors_used']
            })
        
        return pd.DataFrame(summary_data)
    
    def find_similar_countries(self, df, target_country, top_n=5):
        """
        Find countries most similar to a target country
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        target_country : str
            Name of target country
        top_n : int
            Number of similar countries to return
            
        Returns:
        --------
        similar_countries : list of tuples
            List of (country_name, similarity_score) tuples
        """
        if 'country' not in self.results:
            print("Country-wise results not available. Run evaluate_both_approaches() first.")
            return None
        
        countries = self.results['country']['countries']
        
        # Find target country index
        if target_country not in countries:
            print(f"Country '{target_country}' not found in dataset.")
            return None
        
        target_idx = np.where(countries == target_country)[0][0]
        
        # Get similarity matrix
        similarity_matrix = self.results['country']['similarity_matrix']
        
        # Get similarities for target country
        similarities = similarity_matrix[target_idx]
        
        # Sort by similarity (descending), excluding self
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = sorted_indices[sorted_indices != target_idx]  # Remove self
        
        # Get top n similar countries
        top_indices = sorted_indices[:top_n]
        
        similar_countries = []
        for idx in top_indices:
            similar_countries.append((countries[idx], similarities[idx]))
        
        print(f"\nTop {top_n} countries most similar to {target_country}:")
        for i, (country, similarity) in enumerate(similar_countries):
            print(f"  {i+1}. {country} (similarity: {similarity:.4f})")
        
        return similar_countries
    
    def find_similar_industries(self, df, target_industry, top_n=5):
        """
        Find industries most similar to a target industry
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        target_industry : str
            Name of target industry (column name)
        top_n : int
            Number of similar industries to return
            
        Returns:
        --------
        similar_industries : list of tuples
            List of (industry_name, similarity_score) tuples
        """
        if 'industry' not in self.results:
            print("Industry-wise results not available. Run evaluate_both_approaches() first.")
            return None
        
        industries = self.results['industry']['industries']
        
        # Find target industry index
        if target_industry not in industries:
            print(f"Industry '{target_industry}' not found in dataset.")
            return None
        
        target_idx = industries.index(target_industry)
        
        # Get similarity matrix
        similarity_matrix = self.results['industry']['similarity_matrix']
        
        # Get similarities for target industry
        similarities = similarity_matrix[target_idx]
        
        # Sort by similarity (descending), excluding self
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = sorted_indices[sorted_indices != target_idx]  # Remove self
        
        # Get top n similar industries
        top_indices = sorted_indices[:top_n]
        
        similar_industries = []
        for idx in top_indices:
            similar_industries.append((industries[idx], similarities[idx]))
        
        print(f"\nTop {top_n} industries most similar to {target_industry}:")
        for i, (industry, similarity) in enumerate(similar_industries):
            print(f"  {i+1}. {industry} (similarity: {similarity:.4f})")
        
        return similar_industries
    
    def analyze_country_performance(self, approach='country', top_n=10):
        """
        Analyze performance by country
        
        Parameters:
        -----------
        approach : str
            'country' or 'industry'
        top_n : int
            Number of top/bottom countries to show
        """
        if approach not in self.results:
            print(f"{approach}-wise results not available.")
            return None
        
        results = self.results[approach]
        countries = results['countries']
        country_indices = results['country_indices']
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Calculate RMSE per country
        country_errors = {}
        country_counts = {}
        
        for idx, country_idx in enumerate(country_indices):
            country_name = countries[country_idx]
            error = (actuals[idx] - predictions[idx]) ** 2
            
            if country_name not in country_errors:
                country_errors[country_name] = 0
                country_counts[country_name] = 0
            
            country_errors[country_name] += error
            country_counts[country_name] += 1
        
        # Compute RMSE per country
        country_rmse = {}
        for country in country_errors:
            country_rmse[country] = np.sqrt(country_errors[country] / country_counts[country])
        
        # Sort countries by RMSE
        sorted_countries = sorted(country_rmse.items(), key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"COUNTRY-WISE PERFORMANCE ANALYSIS ({approach.upper()}-wise k-NN)")
        print('='*70)
        
        print(f"\nTop {top_n} countries with LOWEST RMSE (best predictions):")
        for i, (country, rmse) in enumerate(sorted_countries[:top_n]):
            print(f"  {i+1:2d}. {country:30s} RMSE: {rmse:.6f}")
        
        print(f"\nTop {top_n} countries with HIGHEST RMSE (worst predictions):")
        for i, (country, rmse) in enumerate(sorted_countries[-top_n:][::-1]):
            print(f"  {i+1:2d}. {country:30s} RMSE: {rmse:.6f}")
        
        return country_rmse
    
    def analyze_industry_performance(self, approach='country', top_n=10):
        """
        Analyze performance by industry
        
        Parameters:
        -----------
        approach : str
            'country' or 'industry'
        top_n : int
            Number of top/bottom industries to show
        """
        if approach not in self.results:
            print(f"{approach}-wise results not available.")
            return None
        
        results = self.results[approach]
        industries = results['industries']
        industry_indices = results['industry_indices']
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Calculate RMSE per industry
        industry_errors = {}
        industry_counts = {}
        
        for idx, industry_idx in enumerate(industry_indices):
            industry_name = industries[industry_idx]
            error = (actuals[idx] - predictions[idx]) ** 2
            
            if industry_name not in industry_errors:
                industry_errors[industry_name] = 0
                industry_counts[industry_name] = 0
            
            industry_errors[industry_name] += error
            industry_counts[industry_name] += 1
        
        # Compute RMSE per industry
        industry_rmse = {}
        for industry in industry_errors:
            industry_rmse[industry] = np.sqrt(industry_errors[industry] / industry_counts[industry])
        
        # Sort industries by RMSE
        sorted_industries = sorted(industry_rmse.items(), key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"INDUSTRY-WISE PERFORMANCE ANALYSIS ({approach.upper()}-wise k-NN)")
        print('='*70)
        
        print(f"\nTop {top_n} industries with LOWEST RMSE (best predictions):")
        for i, (industry, rmse) in enumerate(sorted_industries[:top_n]):
            print(f"  {i+1:2d}. {industry:20s} RMSE: {rmse:.6f}")
        
        print(f"\nTop {top_n} industries with HIGHEST RMSE (worst predictions):")
        for i, (industry, rmse) in enumerate(sorted_industries[-top_n:][::-1]):
            print(f"  {i+1:2d}. {industry:20s} RMSE: {rmse:.6f}")
        
        return industry_rmse
    
    def export_detailed_predictions(self, df, approach='country', filename=None):
        """
        Export detailed predictions to CSV
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input dataframe
        approach : str
            'country' or 'industry'
        filename : str
            Output filename (optional)
        """
        if approach not in self.results:
            print(f"{approach}-wise results not available.")
            return None
        
        results = self.results[approach]
        
        # Create detailed results DataFrame
        detailed_df = pd.DataFrame({
            'Country': df['Country'].iloc[results['country_indices']].values,
            'Industry': [results['industries'][i] for i in results['industry_indices']],
            'Actual_Value': results['actuals'],
            'Predicted_Value': results['predictions'],
            'Absolute_Error': np.abs(results['actuals'] - results['predictions']),
            'Relative_Error': np.abs((results['actuals'] - results['predictions']) / 
                                   (results['actuals'] + 1e-10)),  # Avoid division by zero
            'Neighbors_Used': results['neighbors_used']
        })
        
        # Add error percentage
        detailed_df['Error_Percentage'] = detailed_df['Relative_Error'] * 100
        
        if filename:
            detailed_df.to_csv(filename, index=False)
            print(f"Detailed predictions saved to {filename}")
        
        return detailed_df
    
    def visualize_similarity_matrix(self, approach='country', top_n=20):
        """
        Visualize similarity matrix (heatmap)
        
        Parameters:
        -----------
        approach : str
            'country' or 'industry'
        top_n : int
            Number of items to show (for readability)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if approach not in self.results:
            print(f"{approach}-wise results not available.")
            return None
        
        results = self.results[approach]
        similarity_matrix = results['similarity_matrix']
        
        # Get labels
        if approach == 'country':
            labels = results['countries']
            title = "Country Similarity Matrix (Cosine Similarity)"
        else:
            labels = results['industries']
            title = "Industry Similarity Matrix (Cosine Similarity)"
        
        # Limit to top_n for readability
        n_show = min(top_n, len(labels))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix[:n_show, :n_show],
                   xticklabels=labels[:n_show],
                   yticklabels=labels[:n_show],
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f"{title} (First {n_show} items)", fontsize=14)
        plt.xlabel(f"{approach.capitalize()}s", fontsize=12)
        plt.ylabel(f"{approach.capitalize()}s", fontsize=12)
        plt.tight_layout()
        plt.show()


# Convenience function for quick analysis
def run_knn_analysis(df, k_neighbors=5, scale_data=True):
    """
    Run complete k-NN analysis pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Your carbon emissions dataframe
    k_neighbors : int
        Number of nearest neighbors
    scale_data : bool
        Whether to scale data
        
    Returns:
    --------
    knn_model : KNNCollaborativeFiltering object
    all_results : dict
        Results from both approaches
    summary_df : pandas DataFrame
        Summary of results
    """
    # Initialize model
    knn_model = KNNCollaborativeFiltering(k_neighbors=k_neighbors, scale_data=scale_data)
    
    # Run evaluation
    all_results = knn_model.evaluate_both_approaches(df)
    
    # Get summary
    summary_df = knn_model.get_summary_dataframe()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    return knn_model, all_results, summary_df


# Function to find optimal k value
def find_optimal_k(df, k_values=None, approach='country', scale_data=True):
    """
    Find optimal k value by testing multiple values
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    k_values : list
        List of k values to test
    approach : str
        'country' or 'industry'
    scale_data : bool
        Whether to scale data
        
    Returns:
    --------
    results_df : pandas DataFrame
        Results for different k values
    best_k : int
        Optimal k value
    """
    if k_values is None:
        k_values = [3, 5, 7, 10, 15, 20]
    
    print(f"Testing k values for {approach}-wise k-NN: {k_values}")
    
    k_results = []
    
    for k in k_values:
        print(f"\nTesting k = {k}")
        print("-" * 30)
        
        knn_model = KNNCollaborativeFiltering(k_neighbors=k, scale_data=scale_data)
        results = knn_model.leave_one_out_cross_validation(df, approach=approach)
        
        k_results.append({
            'k': k,
            'RMSE': results['rmse'],
            'R²': results['r2'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape'],
            'Avg Neighbors Used': results['avg_neighbors_used']
        })
    
    results_df = pd.DataFrame(k_results)
    
    # Find best k based on RMSE
    best_k_row = results_df.loc[results_df['RMSE'].idxmin()]
    best_k = best_k_row['k']
    
    print("\n" + "=" * 70)
    print(f"OPTIMAL k SELECTION FOR {approach.upper()}-WISE APPROACH")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nOptimal k value: {best_k} (lowest RMSE: {best_k_row['RMSE']:.6f})")
    
    return results_df, best_k