import pytest
import os
import json
from app.config_handler import load_config, save_config, merge_config
from app.plugin_loader import load_plugin

def setup_module(module):
    if not os.path.exists('tests/config'):
        os.makedirs('tests/config')

def teardown_module(module):
    if os.path.exists('tests/config/test_config.json'):
        os.remove('tests/config/test_config.json')
    if os.path.exists('config_out.json'):
        os.remove('config_out.json')

# Test loading and saving configuration
def test_load_save_config():
    config = {'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv', 'plugin': 'default_plugin'}
    save_config(config, 'tests/config/test_config.json')
    loaded_config = load_config(type('', (), {'load_config': 'tests/config/test_config.json'})())
    assert loaded_config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert loaded_config['plugin'] == 'default_plugin'

# Test merging configuration
def test_merge_config():
    initial_config = {'csv_file': 'tests/data/EURUSD_5m_2006_2007.csv'}
    cli_args = {'plugin': 'default_plugin'}
    merged_config = merge_config(initial_config, cli_args)
    assert merged_config['csv_file'] == 'tests/data/EURUSD_5m_2006_2007.csv'
    assert merged_config['plugin'] == 'default_plugin'

# Test loading plugin
def test_load_plugin():
    plugin_class, required_params = load_plugin('default_plugin')
    assert plugin_class is not None
    assert 'method' in required_params
