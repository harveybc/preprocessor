# -*- coding: utf-8 -*-

import pytest
import csv

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class  TestPreprocessor:
    """ Base class for testing the preprocessors """
    
    def __init__(self, conf):
        """ Component Tests Constructor """
        self.input_file = conf.input_file 
        """ Test dataset filename """
        self.output_file = conf.output_file
        """ Output dataset filename """
        self.input_config = conf.input_config_file
        """ Output dataset filename """
        self.output_config = conf.output_config_file
        """ Output configuration of the proprocessor """
        self.rows_d, self.cols_d = self.get_size_csv(self.input_file)
        """ Get the number of rows and columns of the test dataset """

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list( csv.reader(open(csv_file)) )
        return len(rows), len(rows[0])
