import pandas as pd

def extract_ceda_data(excel_file_path):
    """
    Extract CEDA data from row 28 to 177, column C to ON from the Excel file.
    
    Parameters:
    excel_file_path (str): Path to the Excel file
    
    Returns:
    pandas.DataFrame: Extracted data as a DataFrame
    """
    try:
        # Read the Excel file, skipping the first 27 rows (0-indexed, so we skip 27 to start at row 28)
        # Use header=None since we're reading raw data
        df = pd.read_excel(
            excel_file_path, 
            sheet_name='Open CEDA', 
            header=None, 
            skiprows=27,  # Skip first 27 rows to start at row 28
            nrows=150,    # Read 150 rows (from row 28 to 177 inclusive)
            usecols='C,E:ON',  # Use columns C through ON
            dtype=str
        )
        
        # Reset index to have clean row numbers starting from 0
        df.reset_index(drop=True, inplace=True)
        
        # The first row (row 28 from the original file) becomes our column headers
        df.columns = df.iloc[0]
        
        # Remove the first row since it's now our column headers
        df = df.iloc[1:].reset_index(drop=True)

        # Convert all columns except the first one (country names) to numeric
        # Keep the first column (country names) as strings
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # errors='coerce' converts invalid parsing to NaN
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {excel_file_path}")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "data\Open CEDA by Watershed.xlsx"
    
    # Extract the data
    ceda_data = extract_ceda_data(file_path)
    
    if ceda_data is not None:
        print(f"Successfully extracted data with shape: {ceda_data.shape}")
        print("\nFirst few rows:")
        print(ceda_data.head())
        
        # Display column names
        print("\nColumn names:")
        print(ceda_data.columns.tolist()[:20])  # Show first 20 columns
        
        # Save to CSV if needed
        ceda_data.to_csv("extracted_ceda_data.csv", index=False)
        print("\nData saved to extracted_ceda_data.csv")