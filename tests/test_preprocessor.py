# -*- coding: utf-8 -*-

import pytest
import csv

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class  TestPreprocessor:
    """ Base class for testing the preprocessors
    """
    
    def __init__(self, test_file, out_file):
    """ Component Tests Constructor
    """
        self.test_file = test_file 
        """ Test dataset filename """
        self.out_file = out_file
        """ Output dataset filename """
        self.out_config = out_config
        """ Output configuration of the proprocessor """
        self.rows_d, self.cols_d = self.get_size_csv(test_file)
        """ Number of rows and columns of the test dataset """

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list( csv.reader(open(csv_file)) )
        return len(rows), len(rows[0])
