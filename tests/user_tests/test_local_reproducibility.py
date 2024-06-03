import os
import subprocess
import pandas as pd

def test_local_reproducibility():
    input_file = 'tests/datasets/EURUSD_5m_2006_2007.csv'
    output_file_1 = 'tests/datasets/output_1.csv'
    output_file_2 = 'tests/datasets/output_2.csv'
    config_file = 'tests/config_out.json'
    
    # Run the preprocessor with the initial parameters
    cmd1 = [
        'python', 'preprocessor', input_file, '--plugin', 'feature_selector',
        '--method', 'select_single', '--single', '2',
        '--output_file', output_file_1,
        '--save_config', config_file
    ]
    subprocess.run(cmd1, check=True)

    # Run the preprocessor again, loading the configuration from the saved file
    cmd2 = [
        'python', 'preprocessor', input_file,
        '--load_config', config_file,
        '--output_file', output_file_2
    ]
    subprocess.run(cmd2, check=True)

    # Load the two output datasets
    df1 = pd.read_csv(output_file_1, header=None)
    df2 = pd.read_csv(output_file_2, header=None)

    # Compare the two output datasets to ensure they are identical
    pd.testing.assert_frame_equal(df1, df2)

    # Clean up test artifacts
    os.remove(output_file_1)
    os.remove(output_file_2)
    os.remove(config_file)

if __name__ == "__main__":
    test_local_reproducibility()
