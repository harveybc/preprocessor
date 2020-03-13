# -*- coding: utf-8 -*-


import pytest
import csv
import sys
import os
from filecmp import cmp

from preprocessor.standardizer.standardizer import Standardizer

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"


class Conf:
    def __init__(self):
        """ Component Tests Constructor """
        fname = os.path.join(os.path.dirname(__file__), "../data/test_input.csv")
        self.input_file = fname
        """ Test dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "../data/test_output.csv")
        self.output_file = fname
        """ Output dataset filename """
        # fname = os.path.join(os.path.dirname(__file__), "../data/in_config.csv")
        # self.input_config_file = fname
        #""" Input configuration of the proprocessor """
        fname = os.path.join(os.path.dirname(__file__), "../data/out_config.csv")
        self.output_config_file = fname
        """ Output configuration of the proprocessor """


class TestStandardizer:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = Standardizer(self.conf)
        """ Data standardizer object """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def test_C03T01_standardize(self):
        """ Standardizes all the constant columns and standardizes all rows with consecutive zeroes from start and end by using the standardize_auto method. Execute standardizer with auto_standardize = true.  """
        rows_t, cols_t = self.dt.standardize()
        # save output to file
        self.dt.store()
        # TODO: ASSERT AVERAGE OF AVERAGES OF COLUMNS LESS THAN 1, more than -1?
        assert (rows_o + cols_o) == (self.rows_d + self.cols_d) - (rows_t + cols_t)

    def test_C03T02_cmdline_standarize(self):
        """ Standardizes all the constant columns using command line arguments """
        os.system(
            "standardizer --remove_columns --no_auto_standardize --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert ((cols_d - cols_o) > 0) and ((cols_o > 0) and (rows_o > 0))

    def test_C03T03_config_save(self):
        """ Save a configuration file and uses it to standardize a dataset. Assert that output_config can be loaded and the output_config(loaded) == output_config(saved)"""
        os.system(
            "standardizer --input_file --from_start 20"
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file 
        )
        # Uses the output as input for another dataset and compare with desired output.
        os.system(
            "standardizer --input_file "
            + self.conf.input_file
            + " --input_config_file "
            + self.conf.output_config_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file  + ".c03t03"
        )
        assert cmp(self.conf.output_config_file, self.conf.output_config_file  + ".c02t07", shallow=True)

    def test_C03T04_config_load(self):
        """ Load a configuration file and uses it to standardize a dataset. Verify that output_config == input_config"""
        fname = os.path.join(os.path.dirname(__file__), "../data/in_config.csv")
        input_config_file = fname
        """ Input configuration of the proprocessor """
        os.system(
            "standardizer --input_file "
            + self.conf.input_file
            + " --input_config_file "
            +  input_config_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file
        )
        assert cmp(input_config_file, self.conf.output_config_file, shallow=True)
