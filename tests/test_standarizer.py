# -*- coding: utf-8 -*-

import pytest
import csv 
import sys
sys.path.append('..\\src\\')
import numpy as np
from data_trimmer.data_trimmer import DataTrimmer 
from test_preprocessor import TestPreprocessor 

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

class  TestDataTrimmer(): 
    """ Component Tests  """
    
    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.st = Standarizer(self.conf)
        """ Data trimmer object """
        self.rows_d, self.cols_d = self.get_size_csv(self.conf.input_file)
        """ Get the number of rows and columns of the test dataset """
    
    def test_C02T01_trim_standarize(self):
        """ Standarize a dataset """
        rows_t, cols_t = self.st.standarize()
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

    