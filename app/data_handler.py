import pandas as pd

def load_csv(filepath, headers=True):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        headers (bool): Whether the first row is headers.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if headers:
        return pd.read_csv(filepath, index_col=0)
    else:
        return pd.read_csv(filepath, header=None, index_col=0)

def write_csv(filepath, data, headers=True, force_date=True):
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        filepath (str): Path to save the CSV file.
        data (pd.DataFrame): DataFrame to write.
        headers (bool): Whether to write headers.
        force_date (bool): Whether to include the date column.
    """
    if not force_date:
        data = data.drop(data.columns[0], axis=1)  # Drop the date column

    if headers:
        data.to_csv(filepath, header=True)
    else:
        data.to_csv(filepath, header=False)
