# -*- coding: utf-8 -*-
"""
This File contains the DataTrimmer class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        data-trimmer = data_trimmer.data_trimmer:run

Then run `python setup.py install` which will install the command `data-trimmer`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

TODO: VERIFICAR

"""

import argparse
import sys
import logging
import numpy as np
from preprocessor.preprocessor import Preprocessor

from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class  DataTrimmer(Preprocessor):
    """ The Data Trimmer preprocessor class """
    
    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        super().__init__(conf)
       
    def parse_args(self, args):
        """ Parse command line parameters

        Args:
            args ([str]): command line parameters as list of strings

        Returns:
            :obj:`argparse.Namespace`: command line parameters namespace
        """
        parser = argparse.ArgumentParser(description="Dataset Trimmer: trims constant columns and consecutive zero rows from the end and the start of a dataset.")
        parser.add_argument("--version", action="version", version="preprocessor {ver}".format(ver=__version__))
        parser.add_argument("--input_file", help="Input CSV filename ",type=string)
        parser.add_argument("--output_file",help="Output CSV filename",type=string)
        parser.add_argument("--input_config_file",help="Input configuration  filename")
        parser.add_argument("--output_config_file",help="Output configuration  filename",type=string)
        parser.add_argument("--from_start", help="number of rows to remove from start (ignored if auto_trim)",type=int, default = 0)
        parser.add_argument("--from_end",help="number of rows to remove from end (ignored if auto_trim)",type=int, default = 0)
        parser.add_argument("--remove_columns",help="removes constant columns",action='store_true')
        parser.add_argument("--auto_trim",help="trims the constant columns and trims all rows with consecutive zeroes from start and end",action='store_true')
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser.parse_args(args)

    def core(self, args):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """
        if (args.auto_trim):
            self.trim_auto()
        elif (args.remove_columns):
            self.trim_columns()
        elif (args.from_start>0) and (args.from_end>0):
            self.trim_fixed_rows(args.from_start, args.from_end)
        else:
            _logger.info("Error in command-line parameter...")

    def trim_fixed_rows(self, from_start, from_end):
        """ Trims a configurable number of rows from the start or end of the input dataset

        Args:
            from_start (int): number of rows to remove from start (ignored if auto_trim)
            from_end (int): number of rows to remove from end (ignored if auto_trim)

        Returns:
            rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        # remove from start
        self.output_ds = self.input_ds[from_start:len(self.input_ds),:]
        # remove from end
        self.output_ds = self.output_ds[:len(self.output_ds)-from_end,:]
        return from_end+from_start, 0
 
    def trim_columns(self):
        """ Trims all the constant columns from the input dataset

        Returns:
            rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        self.rows_d, self.cols_d = self.input_ds.shape
        # initialize unchanged_array as true with size num_columns
        un_array = [True] * self.cols_d
        # in two consecutive rows, search the unchanged values
        for i in range(self.rows_d-1):
            unchanged = (self.input_ds[i, :] == self.input_ds[i+1, :]).all()
            # for each un_array that is true, if the values changed, set it to false
            un_array = not (un_array and unchanged)
        # remove all rows with true on the un_array
        self.output_ds=self.input_ds[:,not(un_array)]
        return 0, np.sum(un_array)

    def trim_auto(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end of the input dataset
 
        Returns:
        rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        self.rows_d, self.cols_d = self.input_ds.shape
        rows_t, cols_t = self.trim_columns()
        # delete rows from start that contain zeroes from start
        z_array = (self.output_ds[0]==0)
        while (np.any(z_array)):
            rows_t = rows_t + 1
            # delete the first row of the output_ds and updates z_array
            self.output_ds = np.delete(self.output_ds, [0], axis=0)
            z_array = (self.output_ds[0]==0)
        return rows_t, cols_t

    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        np.savetxt(self.input_file, self.output_ds, delimiter=",")

def run():
    """ Entry point for console_scripts """
    data_trimmer = DataTrimmer(None)
    data_trimmer.main(sys.argv[1:])


if __name__ == "__main__":
    run()
