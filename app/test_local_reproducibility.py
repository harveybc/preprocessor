import os
import subprocess
import filecmp

def test_local_reproducibility():
    # Define the initial command
    initial_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--plugin', 'feature_selector',
        '--method', 'select_single',
        '--single', '2',
        '--output_file', 'tests/data/output_1.csv',
        '--save_config', 'tests/config_out.json'
    ]

    # Run the initial command
    subprocess.run(initial_command, check=True)

    # Define the second command with --load_config
    second_command = [
        'python', '-m', 'app.main',
        'tests/data/EURUSD_5m_2006_2007.csv',
        '--load_config', 'tests/config_out.json',
        '--output_file', 'tests/data/output_2.csv'
    ]

    # Run the second command
    subprocess.run(second_command, check=True)

    # Compare the two output files
    assert filecmp.cmp('tests/data/output_1.csv', 'tests/data/output_2.csv'), "The two output files are not identical."

if __name__ == '__main__':
    test_local_reproducibility()
