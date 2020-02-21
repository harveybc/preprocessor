# -*- coding: utf-8 -*-
"""
 To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = data_trimmer.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

"""

import argparse
import sys
import logging

from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class  Preprocessor:
    """ Component Tests
    """
    
    def __init__(self):
    """ Component Tests Constructor
    """
        self.input_file = ""
        """ Path of the test dataset """    
        self.output_file = ""
        """ Path of the output dataset """    
        self.input_config = ""
        """ Path of the input configuration """    
        self.output_config = ""
        """ Path of the output configuration """    
        self.rows_d, self.cols_d = self.get_size_csv(test_file)
        """ Number of rows and columns in the test dataset """

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.
        
        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list( csv.reader(open(csv_file)) )
        return len(rows), len(rows[0])

    def setup_logging(self, loglevel):
        """Setup basic logging

        Args:
        loglevel (int): minimum loglevel for emitting messages
        """
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(level=loglevel, stream=sys.stdout,
                            format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

    def main(self, args):
        """Main entry point allowing external calls

        Args:
        args ([str]): command line parameter list
        """
        args = self.parse_args(args)
        self.setup_logging(args.loglevel)
        _logger.debug("Starting crazy calculations...")
        print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
        _logger.info("Script ends here")

    def parse_args(self, args):
        """Parse command line parameters, to be overriden by child classes depending on their command line parameters if they are console scripts.

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    pass 

