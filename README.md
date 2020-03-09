# Preprocessor

A simple data pre-processor. Usable both from command line and from class methods.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)
Dataset Trimmer


## Description

Trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes or a fixed number of rows. Save a CSV with removed files and columns for applying similar  trimming to another dataset. Usable both from command line and from class methods (see [tests folder](https://github.com/harveybc/preprocessor/tree/master/tests)).

## Installation

For now the installation is made by clonning the github repo and manually installing it, package based installation comming soon..

### Steps
1. Clone the GithHb repo:   
> git clone https://github.com/harveybc/preprocessor
2. Change to the repo folder:
> cd preprocessor
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. Perform tests
> python setup.py install

## Data-Trimmer

The data-trimmer is implemented in the DataTrimmer class, it has methods for loading a dataset trimming it an producing an  output, please see [test_data_trimmer], tests 1 to 3.

### Command-Line Execution

The data-trimmer also is implemented as a console command:
> data-trimmer -- input_file <input_dataset> <optoional_parameters>

### Command-Line Parameters

* __--input_file__: The only mandatory parameter, is the filename for the input dataset to be trimmed.

