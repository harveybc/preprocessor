# -*- coding: utf-8 -*-
""" This File contains the Preprocessor class, it is the base class for DataTrimmer, FeatureSelector, Standarizer, MSSADecomposer. """

import argparse
import sys
import logging

from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class  Preprocessor:
    """ Base class for DataTrimmer, FeatureSelector, Standarizer, MSSADecomposer. """
    
    def __init__(self):
    """ Constructor """
        self.input_file = ""
        """ Path of the input dataset """    
        self.output_file = ""
        """ Path of the output dataset """    
        self.input_config_file = ""
        """ Path of the input configuration """    
        self.output_config_file = ""
        """ Path of the output configuration """   
        self.args = None
        self.rows_d, self.cols_d = self.get_size_csv(input_file)
        """ Number of rows and columns in the test dataset """
        self.input_ds = None
        """ Input dataset """ 
        self.output_ds = None
        """ Output dataset """ 
        self.output_config = None
        """ Output configuration """ 
        
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
        """Setup basic logging.

        Args:
        loglevel (int): minimum loglevel for emitting messages
        """
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(level=loglevel, stream=sys.stdout,
                            format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

    def main(self, args):
        """ Starts an instance. Main entry point allowing external calls.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        args = self.parse_args(args)
        # Start logging
        self.setup_logging(args.loglevel)
        _logger.info("Starting preprocessor...")
        # Load input dataset
        self.input_ds = list( csv.reader( open(csv_file) ) )
        # Start core function
        self.core(args)
        _logger.debug("Saving results...")
        # Save results and output configuration
        self.store(args)
        _logger.info("Script end.")

    def store(self)
        """ Save preprocessed data and the configuration of the preprocessor. """
        pass

    def core(self, args)
        """ Core preprocessor task after starting the instance with the main method.
            To be overriden by child classes depending on their preprocessor task.

        Args:
        args (obj): command line parameters as objects
        """
        pass

    def parse_args(self, args):
        """Parse command line parameters, to be overriden by child classes depending on their command line parameters if they are console scripts.

        Args:
        args ([str]): command line parameters as list of strings

        Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
        """
        pass 

