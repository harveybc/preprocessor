import subprocess
import json
import os
import requests
import pytest

# Setup and teardown functions
def setup_module(module):
    if not os.path.exists('tests/data'):
        os.makedirs('tests/data')
    if not os.path.exists('tests/config'):
        os.makedirs('tests/config')

def teardown_module(module):
    if os.path.exists('config_out.json'):
        os.remove('config_out.json')
    if os.path.exists('debug_out.json'):
        os.remove('debug_out.json')
    if os.path.exists('tests/data/output.csv'):
        os.remove('tests/data/output.csv')

# Test to check if CLI arguments are parsed correctly and the default plugin is used
def test_cli_args_and_default_plugin():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv'
    ]
    subprocess.run(command, check=True)

    with open('config_out.json', 'r') as f:
        config = json.load(f)
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert config['plugin'] == 'default_plugin'

    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "min_val", "max_val"}
    assert set(debug_info.keys()).issuperset(expected_debug_keys)

# Test configuration handling by loading a pre-existing config and saving a new one
def test_config_handling():
    # Save an initial config
    initial_config = {
        'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv',
        'plugin': 'feature_selector',
        'method': 'select_single',
        'single': 3
    }
    with open('tests/config/test_config.json', 'w') as f:
        json.dump(initial_config, f)

    command = [
        'python', '-m', 'app.main',
        '--load_config', 'tests/config/test_config.json'
    ]
    subprocess.run(command, check=True)

    with open('config_out.json', 'r') as f:
        config = json.load(f)
    assert config['csv_file'] == initial_config['csv_file']
    assert config['plugin'] == initial_config['plugin']
    assert config['method'] == initial_config['method']
    assert config['single'] == initial_config['single']

# Test data handling by loading, processing, and writing CSV data
def test_data_handling():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--output_file', 'tests/data/output.csv'
    ]
    subprocess.run(command, check=True)

    assert os.path.exists('tests/data/output.csv')

    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)
    assert debug_info['output_rows'] > 0
    assert debug_info['output_columns'] > 0

# Test plugin loading and execution
def test_plugin_loading_and_execution():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'feature_selector',
        '--single', '3'
    ]
    subprocess.run(command, check=True)

    with open('config_out.json', 'r') as f:
        config = json.load(f)
    assert config['plugin'] == 'feature_selector'
    assert config['single'] == 3

    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns", "method", "single"}
    assert set(debug_info.keys()).issuperset(expected_debug_keys)

# Test remote operations: save, load, and log
def test_remote_operations():
    save_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_save_config', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(save_command, check=True)

    load_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_load_config', 'http://localhost:60500/preprocessor/feature_selector/detail/1',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(load_command, check=True)

    log_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_log', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(log_command, check=True)

    response = requests.get('http://localhost:60500/preprocessor/feature_selector/detail/1')
    response_data = response.json()
    assert response.status_code == 200
    assert 'result' in response_data

if __name__ == '__main__':
    pytest.main()
