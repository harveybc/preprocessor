# -*- coding: utf-8 -*-


import pytest
import csv
import sys
import os
sys.path.append('..\\src\\')
from src.data_trimmer.data_trimmer import DataTrimmer

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


class TestDataTrimmer():
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = DataTrimmer(self.conf)
        """ Data trimmer object """
        self.rows_d, self.cols_d = self.get_size_csv(self.conf.input_file)
        """ Get the number of rows and columns of the test dataset """

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.

        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list(csv.reader(open(csv_file)))
        return (len(rows), len(rows[0]))

    def test_C02T01_trim_fixed_rows(self):
        """ Trims a configurable number of rows from the start or end of the input dataset by using the trim_fixed_rows method. Execute trimmer with from_start=10, from_end=10. """
        rows_t, cols_t = self.dt.trim_fixed_rows(10, 10)
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d +
                                     self.cols_d) - (rows_t + cols_t)

    def test_C02T02_trim_columns(self):
        """ Trims all the constant columns by using the trim_columns method. Execute trimmer with remove_colums = true. """
        rows_t, cols_t = self.dt.trim_columns()
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d +
                                     self.cols_d) - (rows_t + cols_t)

    def test_C02T03_trim_auto(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end by using the trim_auto method. Execute trimmer with auto_trim = true.  """
        rows_t, cols_t = self.dt.trim_auto()
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d +
                                     self.cols_d) - (rows_t + cols_t)

    def test_C02T04_cmdline(self):
        """ Trims all the constant columns and 10  rows from start and end using command line arguments """
        os.system(
            'data-trimmer --from_start=10 --from_end=10 --remove_colums --file_input=test_input --file_output=file_output ')
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert ((rows_d - rows_o) > 0) and ((cols_d - cols_o)
                                            > 0) and (cols_o > 0) and (rows_o)
