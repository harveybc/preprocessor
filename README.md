# Preprocessor

A simple data pre-processor. Usable both from command line and from class methods.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)
Dataset Trimmer


## Description

Trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes or a fixed number of rows. Save a CSV with removed files and columns for applying similar  trimming to another dataset. Usable both from command line and from class methods (see [tests folder](https://github.com/harveybc/preprocessor/tree/master/tests)).


## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
