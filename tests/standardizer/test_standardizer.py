# -*- coding: utf-8 -*-

import pytest
import csv 
import sys
import numpy as np
#sys.path.append('..\\src\\')
from preprocessor.standardizer.standardizer import Standardizer


__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    def __init__(self):
        """ Component Tests Constructor """
        self.input_file = "tests\\test_input.csv"
        """ Test dataset filename """
        self.output_file = "tests\\test_output.csv"
        """ Output dataset filename """
        self.input_config_file = "in_config.csv"
        """ Output dataset filename """
        self.output_config_file = "out_config.csv"
        """ Output configuration of the proprocessor """

class  TestStandardizer(): 
    """ Component Tests  """
    
    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.st = Standardizer(self.conf)
        """ Data trimmer object """
    
    def atest_C02T01_standarize(self):
        """ Standarize a dataset """
        self.st.standarize()
        # save the output dataset
        self.st.store()
        # read the standarized dataset 
        out_ds = np.genfromtxt(self.output_file, delimiter=',')
        # calculate the average of each column of th3 output dataset
        avg = out_ds.mean(axis=1)
        # calculate the average of the column averages
        avg_cols = avg.mean(axis=0)
        # assert if the average of the averages per column is less than 2
        assert (avg_cols[0] < 2)

    