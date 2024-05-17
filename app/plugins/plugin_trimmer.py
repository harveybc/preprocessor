import pandas as pd
import os
import json

class Plugin:
    def __init__(self):
        # Initialize the trimmer parameters to None
        self.trimmer_params = None

    def process(self, data, columns=None, files=None, save_params=None, load_params=None):
        """
        Remove specified columns from the data and specified files from the filesystem.

        Args:
            data (pd.DataFrame): The input data to be processed.
            columns (list): List of columns to remove from the data.
            files (list): List of file paths to remove from the filesystem.
            save_params (str): Path to save the trimmer parameters.
            load_params (str): Path to load the trimmer parameters.

        Returns:
            pd.DataFrame: The trimmed data.
        """
        # Load parameters if load_params path is provided
        if load_params and os.path.exists(load_params):
            with open(load_params, 'r') as f:
                self.trimmer_params = json.load(f)
            # Load the columns and files to be removed from the loaded parameters
            columns = self.trimmer_params.get('columns', columns)
            files = self.trimmer_params.get('files', files)

        # Save the provided parameters if they are not already loaded
        if self.trimmer_params is None:
            self.trimmer_params = {'columns': columns, 'files': files}
            if save_params:
                with open(save_params, 'w') as f:
                    json.dump(self.trimmer_params, f)

        # Remove specified columns from the DataFrame
        if columns:
            for column in columns:
                if column in data.columns:
                    data.drop(columns=column, inplace=True)
                else:
                    print(f"Warning: Column '{column}' not found in the data.")

        # Remove specified files from the filesystem
        if files:
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
                else:
                    print(f"Warning: File '{file}' not found.")

        return data
