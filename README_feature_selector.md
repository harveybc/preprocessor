# Preprocessor: Feature Selector

Performs feature selection based on a classification or regression training signal and a threshold. 

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)
[![Discord Chat](https://img.shields.io/discord/701635039678562345.svg)](https://discord.gg/NRQw9Cy)  

## Description

Performs the feature selection based on a classification or regression training signal and a threeshold. 

Usable both from command line and from class methods.

The feature selector is implemented in the FeatureSelector class, it has methods for loading a dataset, performing feature selection via the feature_select() method for producing an output dataset, please see [test_feature_selector.py](https://github.com/harveybc/preprocessor/blob/master/tests/feature_selector/test_feature_selector.py). 

## Installation

The module is installed with the preprocessor package, the instructions are described in the [preprocessor README](../master/README.md).

### Command-Line Execution

The feature selection is also implemented as a console command:
> feature_selector -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: Mandatory parameter, is the filename for the input dataset to be processed.
* __--training_file <filename>__: (Optional) Mandatory parameter, is the filename for the training dataset.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--percentile <int>__: (Optional) Feature selection percentile of features to keep, defaults to 20.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.feature_selector.feature_selector import FeatureSelector
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
        self.training_file = "tests/data/test_training.csv"
conf = Conf()
# instance trimmer class and loads dataset
st = FeatureSelector(conf)
# perform the core method
st.core()
# save output to output file
st.store()
```

### Usage via CLI

> feature_selector --input_file "tests/data/test_input.csv" --training_file "tests/data/test_training.csv"






