import pytest
from app.cli import parse_args
from app.config_handler import load_config, save_config, merge_config, save_debug_info
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_plugin
import os
import json
import pandas as pd

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

# Unit test for CLI argument parsing
def test_parse_args(monkeypatch):
    monkeypatch.setattr('sys.argv', ['program_name', 'tests/data/EURUSD_5m_2006_2007.csv', '--plugin', 'default_plugin'])
    args, unknown = parse_args()
    assert args.csv_file == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert args.plugin == 'default_plugin'

# Unit test for loading configuration
def test_load_config():
    initial_config = {'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv'}
    with open('tests/config/test_config.json', 'w') as f:
        json.dump(initial_config, f)
    args = type('', (), {})()  # Create an empty object
    args.load_config = 'tests/config/test_config.json'
    config = load_config(args)
    assert config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'

# Unit test for saving configuration
def test_save_config():
    config = {'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv'}
    save_config(config, 'config_out.json')
    with open('config_out.json', 'r') as f:
        loaded_config = json.load(f)
    assert loaded_config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'

# Unit test for merging configuration
def test_merge_config():
    config = {'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv'}
    cli_args = {'plugin': 'default_plugin'}
    merged_config = merge_config(config, cli_args)
    assert merged_config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert merged_config['plugin'] == 'default_plugin'

# Unit test for saving debug info
def test_save_debug_info():
    debug_info = {
        "execution_time": 0.5,
        "input_rows": 100,
        "output_rows": 100,
        "input_columns": 5,
        "output_columns": 5
    }
    save_debug_info(debug_info, 'debug_out.json')
    with open('debug_out.json', 'r') as f:
        loaded_debug_info = json.load(f)
    assert loaded_debug_info['execution_time'] == 0.5
    assert loaded_debug_info['input_rows'] == 100
    assert loaded_debug_info['output_rows'] == 100
    assert loaded_debug_info['input_columns'] == 5
    assert loaded_debug_info['output_columns'] == 5

# Unit test for loading CSV data
def test_load_csv():
    data = load_csv('tests/data/EURUSD_5m_2006_2007.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

# Unit test for writing CSV data
def test_write_csv():
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    write_csv('tests/data/output.csv', data)
    assert os.path.exists('tests/data/output.csv')

# Unit test for plugin loading
def test_load_plugin():
    plugin_class, required_params = load_plugin('default_plugin')
    assert plugin_class is not None
    assert 'method' in required_params

# Unit tests for individual plugins
def test_default_plugin():
    from app.default_plugin import DefaultPlugin
    plugin = DefaultPlugin()
    assert plugin.params['method'] == 'min-max'
    assert plugin.params['range'] == (-1, 1)
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    processed_data = plugin.process(data)
    assert not processed_data.empty

def test_unbiaser_plugin():
    from app.plugins.plugin_unbiaser import Plugin as UnbiaserPlugin
    plugin = UnbiaserPlugin()
    assert plugin.params['method'] == 'ma'
    assert plugin.params['window_size'] == 5
    data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6]})
    processed_data = plugin.process(data)
    assert not processed_data.empty

def test_feature_selector_plugin():
    from app.plugins.plugin_feature_selector_pre import Plugin as FeatureSelectorPlugin
    plugin = FeatureSelectorPlugin()
    assert plugin.params['method'] == 'select_single'
    assert plugin.params['single'] == 1
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9], 'col4': [10, 11, 12]})
    processed_data = plugin.process(data)
    assert not processed_data.empty

if __name__ == '__main__':
    pytest.main()
