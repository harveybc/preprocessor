# -*- coding: utf-8 -*-

import pytest
import csv
from preprocessor.data_trimmer import DataTrimmer
from tests.test_preprocessor import TestPreprocessor

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class  TestDataTrimmer(TestPreprocessor): 
    """ Component Tests
    """
    
    def __init__(self):
    """ Component Tests Constructor
    """
        super().__init__("test_dataset.csv", "test_output.csv")
        """ Use parent class attributes  
        """
        self.dt = DataTrimmer(self.test_file, self.out_file)
        """ Data trimmer object
        """

    def test_C02T01_trim_rows(self):
        """ Trims a configurable number of rows from the start or end of the output dataset. Execute trimmer with 10 from start and 10 from end.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        # perform trimming using the function trim_data(test_file, out_file, auto_trim=true, column_trim=true, from_start=0, from_end=0)
        rows _t, cols_t = self.dt.trim_data(false, false, 10, 10)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)
        
    def test_C02T02_trim_columns(self):
        """ Must trim all the constant columns. Execute trimmer with remove-colums = true.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        # perform trimming using the function trim_data(test_file, out_file, auto_trim=true, column_trim=true, from_start=0, from_end=0)
        rows_t, cols_t = trim_data(test_file, out_file, false, true)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)

    def test_C02T03_trim_automatic(self):
        """ Must trim all the constant columns. Trim all consecutive zeroes from start and end.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        # perform trimming using the function trim_data(test_file, out_file, auto_trim=true, column_trim=true, from_start=0, from_end=0)
        rows_t, cols_t = trim_data(test_file, out_file)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)