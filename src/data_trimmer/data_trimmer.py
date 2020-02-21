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

from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class  DataTrimmer(Preprocessor):
    """ The Data Trimmer preprocessor class """
    
    def __init__(self):
    """ Constructor using same parameters as base class """
      super().__init__()
       
  def parse_args(args):
      """ Parse command line parameters

      Args:
        args ([str]): command line parameters as list of strings

      Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
      """
      parser = argparse.ArgumentParser(
          description="Dataset Trimmer: trims constant columns and consecutive zero rows from the end and the start of a dataset.")
      parser.add_argument(
          "--version",
          action="version",
          version="preprocessor {ver}".format(ver=__version__))
      parser.add_argument(
          "--from_start",
          help="number of rows to remove from start (ignored if auto_trim)",
          type=int, default = 0)
      parser.add_argument(
          "--from_end",
          help="number of rows to remove from end (ignored if auto_trim)",
          type=int, default = 0)
      parser.add_argument(
          "--remove_columns",
          help="removes constant columns",
          action='store_true')
      parser.add_argument(
          "--auto_trim",
          help="trims the constant columns and trims all rows with consecutive zeroes from start and end",
          action='store_true')
      parser.add_argument(
          "-v",
          "--verbose",
          dest="loglevel",
          help="set loglevel to INFO",
          action="store_const",
          const=logging.INFO)
      parser.add_argument(
          "-vv",
          "--very_verbose",
          dest="loglevel",
          help="set loglevel to DEBUG",
          action="store_const",
          const=logging.DEBUG)
      return parser.parse_args(args)

  def core(self, args)
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
      pass

  def trim_columns(self):
      """ Trims all the constant columns from the input dataset

      Returns:
        rows_t, cols_t (int,int): number of rows and columns trimmed
      """
      pass

def trim_auto(self):
      """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end of the input dataset

      Returns:
        rows_t, cols_t (int,int): number of rows and columns trimmed
      """
      rows_t, cols_t = self.trim_columns()
      
      pass

def run():
    """ Entry point for console_scripts """
    data_trimmer = DataTrimmer()
    data_trimmer.main(sys.argv[1:])


if __name__ == "__main__":
    run()
