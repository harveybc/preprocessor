
# Preprocessor: Unbiaser Moving Average (UnbiaserMA)

A simple data pre-processor that removes the moving average from a dataset and exports the configuration for use on other datasets.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)
[![Discord Chat](https://img.shields.io/discord/701635039678562345.svg)](https://discord.gg/NRQw9Cy)  

## Description

Uses a custom moving average removal method to adjust data points in a dataset by removing the average of the previous N values from each point.

Exports the configuration for use on other datasets.

The unbiaser moving average is implemented in the UnbiaserMA class, which has methods for loading a dataset, processing it to remove the moving average, and producing an output dataset and a configuration file that can be loaded and applied to another dataset. Usable both from command line and from class methods.

## Installation

The module is installed with the preprocessor package, the instructions are described in the [preprocessor README](../master/README.md).

### Command-Line Execution

The unbiaser moving average is also implemented as a console command:
> unbiaser_ma --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be processed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--window_size <size>: (Optional) Number of previous ticks for average calculation. Defaults to 24
## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.unbiaser_ma.unbiaser_ma import UnbiaserMA
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance unbiaser class and loads dataset
um = UnbiaserMA(conf)
# process the dataset to remove the moving average
um.unbias_ma()
# save output to output file
um.store()
```

### Usage via CLI

> unbiaser_ma --input_file "tests/data/test_input.csv"
