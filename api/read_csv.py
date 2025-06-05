import pandas as pd

def read_csv_file(file_path):
    """
    Read and display CSV file content using pandas
    Args:
        file_path (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Display basic information about the dataset
        print("\n1. Basic Information about the Dataset:")
        print("-" * 40)
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")
            
        # Display first 5 rows
        print("\n2. First 5 rows of the dataset:")
        print("-" * 40)
        print(df.head())
        
        # Display basic statistics of numeric columns
        print("\n3. Basic statistics of numeric columns:")
        print("-" * 40)
        print(df.describe())
        
        # Display data types of columns
        print("\n4. Data types of columns:")
        print("-" * 40)
        print(df.dtypes)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return df

if __name__ == "__main__":
    # Example usage
    file_path = "Sample_data_BILLING_EFFICIENCY.csv"  # Replace with your CSV file path
    read_csv_file(file_path) 