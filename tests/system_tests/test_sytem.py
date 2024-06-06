import subprocess
import json
import requests
import pytest

def test_remote_save_config():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_save_config', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(command, check=True)

    response = requests.get('http://localhost:60500/preprocessor/feature_selector/detail/1')
    response_data = response.json()
    assert response.status_code == 200
    assert 'config' in response_data

def test_remote_load_config():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_load_config', 'http://localhost:60500/preprocessor/feature_selector/detail/1',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(command, check=True)

    with open('config_out.json', 'r') as f:
        config = json.load(f)
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'

def test_remote_log():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--remote_log', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    subprocess.run(command, check=True)

    response = requests.get('http://localhost:60500/preprocessor/feature_selector/detail/1')
    response_data = response.json()
    assert response.status_code == 200
    assert 'result' in response_data

def test_default_config():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv'
    ]
    subprocess.run(command, check=True)

    with open('config_out.json', 'r') as f:
        config = json.load(f)
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'

def test_default_debug_info():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv'
    ]
    subprocess.run(command, check=True)

    with open('debug_out.json', 'r') as f:
        debug_info = json.load(f)
    expected_debug_keys = {"execution_time", "input_rows", "output_rows", "input_columns", "output_columns"}
    assert set(debug_info.keys()).issuperset(expected_debug_keys)

if __name__ == '__main__':
    pytest.main()
