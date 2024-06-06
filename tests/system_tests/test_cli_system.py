import subprocess
import os

# Helper function to run CLI commands
def run_cli_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    return result

def test_cli_normalization():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer'
    ]
    run_cli_command(command)
    assert os.path.exists('output.csv')
    assert os.path.exists('config_out.json')
    assert os.path.exists('debug_out.json')

def test_cli_remote_save_config():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer',
        '--remote_save_config', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_cli_command(command)
    # Check if the remote save was successful, this might require an additional request to the server

def test_cli_remote_load_config():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',  # Add the default CSV file here
        '--remote_load_config', 'http://localhost:60500/preprocessor/feature_selector/detail/1',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_cli_command(command)
    assert os.path.exists('output.csv')
    assert os.path.exists('config_out.json')
    assert os.path.exists('debug_out.json')

def test_cli_remote_log():
    command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'normalizer',
        '--remote_log', 'http://localhost:60500/preprocessor/feature_selector/create',
        '--remote_username', 'test',
        '--remote_password', 'pass'
    ]
    run_cli_command(command)
    # Check if the remote log was successful, this might require an additional request to the server