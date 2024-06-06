import subprocess
import json

def test_unbiaser_plugin_default():
    # Define the command to use the unbiaser plugin with default values
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'unbiaser'
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
    assert config['plugin'] == 'unbiaser'
    expected_config_keys = {"csv_file", "plugin"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "method", "window_size", "ema_alpha"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

def test_unbiaser_plugin_window_size_512():
    # Define the command to use the unbiaser plugin with window_size 512
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'unbiaser',
        '--window_size', '512'
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
    assert config['plugin'] == 'unbiaser'
    assert int(config['window_size']) == 512  # Ensure window_size is treated as integer
    expected_config_keys = {"csv_file", "plugin", "window_size"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "method", "window_size", "ema_alpha"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

def test_unbiaser_plugin_ema():
    # Define the command to use the unbiaser plugin with ema method
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'unbiaser',
        '--method', 'ema'
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
    assert config['plugin'] == 'unbiaser'
    assert config['method'] == 'ema'
    expected_config_keys = {"csv_file", "plugin", "method"}
    assert set(config.keys()) == expected_config_keys, f"Unexpected keys in config: {set(config.keys()) - expected_config_keys}"

    # Assertions for the debug file
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "method", "window_size", "ema_alpha"}
    assert set(debug_info.keys()) == expected_debug_keys, f"Unexpected keys in debug info: {set(debug_info.keys()) - expected_debug_keys}"

if __name__ == '__main__':
    test_unbiaser_plugin_default()
    test_unbiaser_plugin_window_size_512()
    test_unbiaser_plugin_ema()
