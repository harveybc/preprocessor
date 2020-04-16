# -*- coding: utf-8 -*-


import pytest
import csv
import sys
import os
import filecmp
import numpy as np

from preprocessor.feature_selector.feature_selector import FeatureSelector

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"


class Conf:
    def __init__(self):
        """ Component Tests Constructor """
        fname = os.path.join(os.path.dirname(__file__), "data/test_input.csv")
        self.input_file = fname
        """ Test dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data\\test_output.csv")
        self.output_file = fname
        """ Output dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data/out_config.csv")
        self.output_config_file = fname
        """ Output configuration of the preprocessor """
        
class TestFeatureSelector:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = FeatureSelector(self.conf)
        # setup training_file for feature selection
        self.dt.training_file = os.path.join(os.path.dirname(__file__), "data/training.csv")
        """ Data feature_selector object """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def test_C04T01_feature_selection(self):
        """ Assert if the output has less columns than the input """        
        self.dt.feature_selection()
        # save output to file
        self.dt.store()
        # read the input and output files
        rows_i = list(csv.reader(open(self.dt.input_file)))
        rows_o = list(csv.reader(open(self.dt.output_file)))
        # perform the assertion 
        assert (len(rows_i[0]) > len(rows_o[0]))
    

    def test_C04T02_cmdline_feature_selection(self):
        """ Assert if the output has less columns than the input using command line arguments """
        os.system(
            "feature_selector --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # read the input and output files
        rows_i = list(csv.reader(open(self.conf.input_file)))
        rows_o = list(csv.reader(open(self.conf.output_file)))
        # perform the assertion 
        assert (len(rows_i[0]) > len(rows_o[0]))
    
    def test_C04T03_config_save_load(self):
        """ Save a configuration file and uses it to feature_selection. Assert that output_config can be loaded and the output_config(loaded) == output_config(saved)"""
        os.system(
            "feature_selector --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file 
        )
        # Uses the output as input for another dataset and compare with desired output.
        os.system(
            "feature_selector --input_file "
            + self.conf.input_file
            + " --input_config_file "
            + self.conf.output_config_file
            + " --output_file "
            + self.conf.output_file + ".c04t03"
        )
        assert filecmp.cmp(self.conf.output_file, self.conf.output_file  + ".c04t03", shallow=True)