import pandas as pd

def load_csv(file_path, headers):
    """
    Load a CSV file into a pandas DataFrame, handling date columns and correct numeric parsing.

    Args:
        file_path (str): The path to the CSV file to be loaded.
        headers (bool): Whether to treat the first row as headers.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        # Read the file with or without headers
        header = 0 if headers else None
        data = pd.read_csv(file_path, header=header, sep=',', parse_dates=[0], dayfirst=True)
        
        # Check if the first column is a date column
        if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
            data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
            data.set_index('date', inplace=True)
        else:
            # Manually set column names if the first column is not a date
            data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # Convert numeric columns to appropriate data types
        for col in data.columns:
            if col != 'date':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
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

def write_csv(file_path, data, headers=True, force_date=True):
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        file_path (str): The path to the CSV file to be written.
        data (pd.DataFrame): The data to be written to the CSV file.
        headers (bool): Whether to write the headers.
        force_date (bool): Whether to include the date column in the output.

    Returns:
        None
    """
    try:
        if not force_date and 'date' in data.columns:
            data = data.drop(columns=['date'])  # Drop the date column if force_date is False
        data.to_csv(file_path, index=True, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise