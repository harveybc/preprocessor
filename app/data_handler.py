import pandas as pd

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame, handling date columns, header detection, and correct numeric parsing.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        # Try to read the file with headers
        data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        
        # Check if the first column is a date column
        if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
            data.set_index(data.columns[0], inplace=True)
    except pd.errors.ParserError:
        # If there is a parsing error, try reading without headers
        data = pd.read_csv(file_path, header=None, sep=',', parse_dates=[0], dayfirst=True)
        data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
        data.set_index('date', inplace=True)

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the file.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise
    
    return data

def write_csv(file_path, data):
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        file_path (str): The path to the CSV file to be written.
        data (pd.DataFrame): The data to be written to the CSV file.

    Returns:
        None
    """
    try:
        data.to_csv(file_path, index=True)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
