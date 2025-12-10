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


def load_sector_metadata(metadata_file_path):
    """
    Load sector metadata from the Excel file and create lookup functions.
    
    Parameters:
    metadata_file_path (str): Path to the metadata Excel file
    
    Returns:
    tuple: (get_name_function, get_description_function) or (None, None) if error
    """
    try:
        # Read the Metadata sheet
        sector_metadata = pd.read_excel(
            metadata_file_path,
            sheet_name='Metadata'
        )
        
        # Create dictionaries for fast lookup
        name_dict = {}
        desc_dict = {}
        
        for _, row in sector_metadata.iterrows():
            sector_code = str(row['Sector Code']).strip()
            name_dict[sector_code] = row['Sector Name']
            desc_dict[sector_code] = row['Industry sector descriptions']
        
        # Define the get_name function
        def get_name(sector_code):
            """Get sector name for a given sector code."""
            return name_dict.get(str(sector_code).strip())
        
        # Define the get_description function
        def get_description(sector_code):
            """Get sector description for a given sector code."""
            return desc_dict.get(str(sector_code).strip())
        
        return get_name, get_description
        
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return None, None


# Example usage:
if __name__ == "__main__":
    # Path to your main data file
    data_file_path = "data\Open CEDA by Watershed.xlsx"
    
    # Path to your metadata file
    metadata_file_path = "data\ceda_metadata.xlsx"  # Update this path
    
    # Extract the CEDA data
    ceda_data = extract_ceda_data(data_file_path)
    
    if ceda_data is not None:
        print(f"Successfully extracted CEDA data with shape: {ceda_data.shape}")
        print("\nFirst few rows:")
        print(ceda_data.head())
        
        # Display column names (these might be sector codes)
        print("\nColumn names (potential sector codes):")
        print(ceda_data.columns.tolist()[:20])
    
    # Load the metadata and get the 2 functions
    get_sector_name, get_sector_description = load_sector_metadata(metadata_file_path)
    
    if get_sector_name and get_sector_description:
        print("\n" + "="*80)
        print("SECTOR LOOKUP FUNCTIONS LOADED")
        print("="*80)
        
        # Test the functions with some sector codes
        test_codes = ['1111A0', '1111B0', '111200', '999999', '']
        
        for code in test_codes:
            print(f"\nTesting sector code: '{code}'")
            
            name = get_sector_name(code)
            if name:
                print(f"  Sector Name: {name}")
                
                # Get description (show first 80 characters)
                desc = get_sector_description(code)
                if desc:
                    print(f"  Description (first 80 chars): {desc[:80]}...")
            else:
                print(f"  Sector code '{code}' not found in metadata.")
        
        # Example: Look up a sector code from your CEDA data columns
        print("\n" + "="*80)
        print("EXAMPLE: LOOKING UP CEDA COLUMN NAMES")
        print("="*80)
        
        if ceda_data is not None:
            # Get some column names from your CEDA data
            sample_columns = ceda_data.columns[:5]  # First 5 columns
            
            for col in sample_columns:
                name = get_sector_name(col)
                if name:
                    print(f"Column '{col}' is: {name}")
                else:
                    print(f"Column '{col}' is not a recognized sector code")