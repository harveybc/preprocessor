import json
import subprocess

def test_normalization_plugin_ma():
    # Define the initial command to normalize with the "ma" method and range (0,1)
    initial_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--method', 'ma',
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
    assert config['output_file'] == 'output.csv'
    assert config['plugin'] == 'default_plugin'
    assert config['method'] == 'ma'
    assert config['range'] == [0, 1]

    # Assertions for the debug file
    assert 'execution_time' in debug_info
    assert debug_info['input_rows'] == 73841
    assert debug_info['output_rows'] == 73841
    assert debug_info['input_columns'] == 5
    assert debug_info['output_columns'] == 5
    assert 'min_val' in debug_info
    assert 'max_val' in debug_info

if __name__ == '__main__':
    test_normalization_plugin_ma()
