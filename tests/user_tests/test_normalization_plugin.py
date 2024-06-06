import subprocess
import json

def test_normalization_plugin():
    # Define the command
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv'
    ]

    # Run the command
    subprocess.run(command, check=True)

    # Load the generated config and debug files
    with open('config_out.json', 'r') as f:
        config = json.load(f)
    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Assertions for the config file
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    expected_config_keys = {"csv_file"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "min_val", "max_val", "mean", "std"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

def test_normalization_plugin_z_score():
    # Define the initial command to normalize with the "z-score" method
    initial_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--method', 'z-score'
    ]

    # Run the initial command
    subprocess.run(initial_command, check=True)

    # Load the generated config and debug files
    with open('config_out.json', 'r') as f:
        config = json.load(f)
    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Assertions for the config file
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['method'] == 'z-score'
    expected_config_keys = {"csv_file", "method"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "min_val", "max_val", "mean", "std"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

def test_normalization_plugin_min_max():
    # Define the initial command to normalize with the "min-max" method and range (0,1)
    initial_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--method', 'min-max',
        '--range', '(0,1)'
    ]

    # Run the initial command
    subprocess.run(initial_command, check=True)

    # Load the generated config and debug files
    with open('config_out.json', 'r') as f:
        config = json.load(f)
    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)

    # Assertions for the config file
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['method'] == 'min-max'
    assert config['range'] == [0, 1]
    expected_config_keys = {"csv_file", "method", "range"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "min_val", "max_val", "mean", "std"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

if __name__ == '__main__':
    test_normalization_plugin()
    test_normalization_plugin_z_score()
    test_normalization_plugin_min_max()
