import os
import json
import pytest
from unittest.mock import patch
from app.main import main

def run_code_based_command(args):
    with patch('sys.argv', args):
        main()

def test_code_normalization():
    args = [
        'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer'
    ]
    run_code_based_command(args)
    assert os.path.exists('output.csv')
    assert os.path.exists('config_out.json')
    assert os.path.exists('debug_out.json')

def test_code_remote_save_config():
    args = [
        'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer',
        '--remote_save_config', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_code_based_command(args)
    # Check if the remote save was successful, this might require an additional request to the server

def test_code_remote_load_config():
    args = [
        'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',  # Add the CSV file here
        '--remote_load_config', 'http://localhost:60500/preprocessor/feature_selector/detail/1',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_code_based_command(args)
    assert os.path.exists('output.csv')
    assert os.path.exists('config_out.json')
    assert os.path.exists('debug_out.json')

def test_code_remote_log():
    args = [
        'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer',
        '--remote_log', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_code_based_command(args)
    # Check if the remote log was successful, this might require an additional request to the server
