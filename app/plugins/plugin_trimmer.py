import pandas as pd
import json
import os

class Plugin:
    def __init__(self):
        # Initialize the trimmer parameters to None
        self.trimmer_params = None

    def process(self, data, columns=None, rows=None, save_params=None, load_params=None):
        """
        Remove specified columns and rows from the dataset.

        Args:
            data (pd.DataFrame): The input data to be processed.
            columns (list): List of column indices to remove from the data.
            rows (list): List of row indices to remove from the data.
            save_params (str): Path to save the trimmer parameters.
            load_params (str): Path to load the trimmer parameters.

        Returns:
            pd.DataFrame: The trimmed data.
        """
        # Load parameters if load_params path is provided
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.trimmer_params = json.load(f)
            # Load the columns and rows to be removed from the loaded parameters
            columns = self.trimmer_params.get('columns', columns)
            rows = self.trimmer_params.get('rows', rows)

        # Save the provided parameters if they are not already loaded
        if self.trimmer_params is None:
            self.trimmer_params = {'columns': columns, 'rows': rows}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.trimmer_params, f)

        # Remove specified columns from the DataFrame
        if columns:
            data.drop(data.columns[columns], axis=1, inplace=True)

        # Remove specified rows from the DataFrame
        if rows:
            data.drop(rows, axis=0, inplace=True)

        return data
