# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
import filecmp
import numpy as np

from preprocessor.sliding_window.sliding_window import SlidingWindow

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    def __init__(self):
        """ Component Tests Constructor """
        fname = os.path.join(os.path.dirname(__file__), "data/test_input.csv")
        self.input_file = fname
        """ Test dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data/test_output.csv")
        self.output_file = fname
        """ Output dataset filename """
        self.window_size = 21
        """ Output configuration of the preprocessor """

class TestSlidingWindow:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = SlidingWindow(self.conf)
        """ Data sliding_window object """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def test_C03T01_window(self):
        """ Perform sliding_window and assert if the output_columns == input_columns * (window_size-1) """        
        self.dt.window_size = self.conf.window_size
        self.dt.sl_window()
        # save output to file
        self.dt.store()
        # read input and output files
        rows_i = list(csv.reader(open(self.conf.input_file)))
        rows_o = list(csv.reader(open(self.conf.output_file)))
        input_columns = len(rows_i[0])
        output_columns = len(rows_o[0])
        assert output_columns == (input_columns * (self.conf.window_size))

    def test_C03T02_cmdline_window(self):
        """ Perform the same C03T01_window assertion but using command line arguments """
        os.system(
            "sliding_window --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --window_size "
            + str(self.conf.window_size)
        )
        rows_i = list(csv.reader(open(self.conf.input_file)))
        rows_o = list(csv.reader(open(self.conf.output_file)))
        input_columns = len(rows_i[0])
        output_columns = len(rows_o[0])
        assert output_columns == (input_columns * (self.conf.window_size))
