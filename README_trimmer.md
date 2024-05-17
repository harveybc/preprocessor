# Trimmer Plugin for Preprocessor

## Description

The Trimmer Plugin is used to remove specified columns and rows from the dataset in the Preprocessor application. This plugin can save and load trimming parameters, making it reusable for consistent preprocessing across different datasets.

## Parameters

The plugin accepts the following parameters:

- **columns** (list): List of column indices to remove from the data.
- **rows** (list): List of row indices to remove from the data.
- **save_params** (str): The file path to save the trimming parameters. Default is `'trimmer_params.json'`.
- **load_params** (str): The file path to load the trimming parameters. If provided, the parameters will be loaded from this file instead of being specified in the command line arguments.

## Usage

### From Command Line

You can use the Trimmer Plugin from the Preprocessor application via command line parameters. Below are some examples:

### Example 1: Removing Specified Columns

```bash
python app/main.py --config config.json --plugin trimmer_plugin --columns 0 1 2
```

### Example 2: Removing Specified Rows

```bash
python app/main.py --config config.json --plugin trimmer_plugin --rows 0 1 2
```

### Example 3: Loading Pre-Saved Trimming Parameters

```bash
python app/main.py --config config.json --plugin trimmer_plugin --load_params trimmer_params.json
```

### Configuration File Example

Here is an example of a JSON configuration file (config.json) that can be used with the Preprocessor application:

```json
{
    "csv_file": "path/to/input.csv",
    "output_file": "path/to/output.csv",
    "plugins": [
        {
            "name": "trimmer_plugin",
            "params": {
                "columns": [0, 1, 2],
                "rows": [0, 1, 2],
                "save_params": "trimmer_params.json",
                "load_params": "trimmer_params.json"
            }
        }
    ],
    "remote_log": "http://remote-log-server/api/logs"
}
```

### Output Configuration File Examples

#### Columns and Rows to Remove (trimmer_params.json)

```json
{
    "columns": [0, 1, 2],
    "rows": [0, 1, 2]
}
```

