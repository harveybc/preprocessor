# -*- coding: utf-8 -*-

import pytest
import csv 
from src.data_trimmer.data_trimmer import DataTrimmer
from tests.test_data_trimmer import TestPreprocessor

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class  TestDataTrimmer(TestPreprocessor): 
    """ Component Tests
    """
    
    def __init__(self):
        """ Component Tests Constructor """
        super().__init__("test_ds.csv", "test_output.csv", "out_config.csv")
        """ Use parent class attributes and test data as parameters for parent class constructor """
        self.dt = DataTrimmer(self.test_file, self.out_file)
        """ Data trimmer object """

    def test_C02T01_trim_fixed_rows(self):
        """ Trims a configurable number of rows from the start or end of the input dataset by using the trim_fixed_rows method. Execute trimmer with from_start=10, from_end=10. """
        rows_t, cols_t = self.dt.trim_fixed_rows(10, 10)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)
        
    def test_C02T02_trim_columns(self):
        """ Trims all the constant columns by using the trim_columns method. Execute trimmer with remove_colums = true. """
        rows_t, cols_t = self.dt.trim_columns()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)

    def test_C02T03_trim_auto(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end by using the trim_auto method. Execute trimmer with auto_trim = true.  """
        rows_t, cols_t = self.dt.trim_auto()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(out_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (rows_d + cols_d) - (rows_t + cols_t)