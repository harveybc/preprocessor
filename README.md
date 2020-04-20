# Preprocessor

A simple timeseries data pre-processor.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)
[![Discord Chat](https://img.shields.io/discord/701635040286867529.svg)](https://discord.gg/NRQw9Cy)  

## Description

Implements modular components for dataset preprocessing: a data-trimmer, a standardizer, a feature selector and a sliding window data generator.

All modules are usable both from command line and from class methods.

## Installation

The installation is made by clonning the github repo and manually installing it, package based installation coming soon..

### Steps
1. Clone the GithHub repo:   
> git clone https://github.com/harveybc/preprocessor
2. Change to the repo folder:
> cd preprocessor
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. (Optional) Perform tests
> python setup.py install
6. (Optional) Generate Sphinx Documentation
> python setup.py docs

## Modules

All the CLI commands and the class modules are installed with the preprocessor package, the following sections describe each module briefly and link to each module's basic documentation. 

Detailed Sphinix documentation for all modules can be generated in HTML format with the optional step 6 of the installation process, it contains documentation of the classes and methods of all modules in the preprocessor package. 

## Data-Trimmer

A simple data pre-processor that trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes. 

See [Data-Trimmer Readme](../master/README_data_trimmer.md) for detailed description and usage instructions.

## Standarizer

Standardizes a dataset and exports the standarization configuration for use on other datasets. 

See [Standardizer Readme](../master/README_standardizer.md) for detailed description and usage instructions.

## Sliding Window

Performs the sliding window technique and exports an expanded dataset with configurable window_size.

See [Sliding Window Readme](../master/README_sliding_window.md) for detailed description and usage instructions.

## Feature Selector

Performs the feature selection based on a classification or regression training signal and a threeshold. 

See [Feature Selector Readme](../master/README_feature_selector.md) for detailed description and usage instructions.

## Examples of usage

The following examples show both the class method and command line uses for one module, for examples of other modules, please see the specific module´s documentation.

### Example: Usage via Class Methods (data_trimmer module)
```python
from preprocessor.data_trimmer.data_trimmer import DataTrimmer
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
dt = DataTrimmer(conf)
# perform the module's core method
dt.core()
# save output to output file
dt.store()
```

### Example: Usage via CLI (data_trimmer module)

> data_trimmer --input_file "tests/data/test_input.csv"






