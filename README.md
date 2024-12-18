
# Preprocessor 

The Preprocessor project is a flexible and modular application for preprocessing time series data. It supports dynamic loading of plugins for various preprocessing tasks such as normalization, unbiasing, trimming, and feature selection. Each plugin can save and load its parameters for consistent preprocessing across different datasets.

## Installation Instructions

To install and set up the Preprocessor application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/preprocessor.git
    cd preprocessor
    ```

2. **Create and Activate a Virtual Environment**:
    - **Using `venv` (Python 3.3+)**:
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        ```

    - **Using `conda`**:
        ```bash
        conda create --name preprocessor_env python=3.9
        conda activate preprocessor_env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **Run the Preprocessor**:
    - On Windows, run the following command to verify installation (it uses all default values, use preprocessor.bat --help, for complete command line arguments description):
        ```bash
        preprocessor.bat 

    - On Linux, run:
        ```bash
        sh preprocessor.sh 
        ```

7. **Run Tests (Optional, requires external repo)**:
For pasing remote tests, requires an instance of [harveybc/data-logger](https://github.com/harveybc/data-logger)
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **Generate Documentation (Optional)**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```

This command should display the help message for the Preprocessor application, confirming that the installation was successful.

## Examples of Use

You can use the Preprocessor application from the command line with various plugins. Below are some examples:

### Default pluigin for preprocessing pipeline with plots

```bash
preprocessor.bat  
```

### Comamnd line parameters:

Use the **-h** or **--help** parameter to get help on the parameters.

### File Structure (outdated):

```md 
preprocessor/
│
├── app/                           # Main application package
│   ├── __init__.py                    # Initializes the Python package
│   ├── main.py                        # Entry point for the application
│   ├── config.py                      # Configuration settings for the app
│   ├── cli.py                         # Command line interface handling
│   ├── data_handler.py                # Module to handle data loading
│   ├── default_plugin.py              # Default plugin (normalizer)
│   └── plugins/                       # Plugins directory
│       ├── __init__.py                # Makes plugins a Python package

│       ├── plugin_unbiaser.py
│       ├── plugin_trimmer.py
│       ├── plugin_feature_selector_pre.py
│       └── plugin_feature_selector_post.py
│
├── tests/                             # Test modules for your application
│   ├── __init__.py                         # Initializes the Python package for tests
│   ├── test_preprocessor.py                # Tests for preprocessor functionality
│   ├── datasets/                           # Test datasets directory
│   └── configs/                            # Test configurations directory
│
├── setup.py                           # Setup file for the package installation
├── README.md                          # Project description and instructions
├── requirements.txt                   # External packages needed
└── .gitignore                         # Specifies intentionally untracked files to ignore

```




