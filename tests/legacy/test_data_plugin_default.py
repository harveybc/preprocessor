import pytest
import pandas as pd
import os
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_plugin

def test_load_csv():
    data = load_csv('tests/data/EURUSD_5m_2006_2007.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_write_csv():
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    write_csv('tests/data/output.csv', data)
    assert os.path.exists('tests/data/output.csv')
    os.remove('tests/data/output.csv')

def test_default_plugin_integration():
    plugin_class, required_params = load_plugin('default_plugin')
    plugin = plugin_class()
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    processed_data = plugin.process(data)
    assert not processed_data.empty
