# -*- coding: utf-8 -*-
""" This File contains the Preprocessor class, it is the base class for DataTrimmer, FeatureSelector, Standardizer, MSSADecomposer. """

import argparse
import sys
import logging
import numpy as np 
import csv
#from preprocessor import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class  Preprocessor:
    """ Base class for DataTrimmer, FeatureSelector, Standardizer, MSSADecomposer. """
    
    def __init__(self, conf):
        """ Constructor """
        # if conf =  None, loads the configuration from the command line arguments
        if conf != None:
            self.input_file = conf.input_file
            """ Path of the input dataset """    
            self.output_file = conf.output_file
            """ Path of the output dataset """    
            self.input_config_file = conf.input_config_file
            """ Path of the input configuration """    
            self.output_config_file = conf.output_config_file
            """ Path of the output configuration """   
            # Load input dataset
            self.load_ds()
        else:
            self.input_ds = None

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
        # Start logging: TODO: Use args.loglevel en lugar de logging.DEBUG
        self.setup_logging(logging.DEBUG)
        _logger.info("Starting preprocessor...")
        # Load input dataset
        if self.input_ds == None: 
            self.load_ds()
        # Start core function
        self.core(args)
        _logger.debug("Saving results...")
        # Save results and output configuration
        self.store()
        _logger.info("Script end.") 

    def load_ds(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        # Load input dataset
        self.input_ds = np.genfromtxt(self.input_file, delimiter=',')
        # Initialize input number of rows and columns
        self.rows_d, self.cols_d = self.input_ds.shape
        


    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        pass

    def core(self, args):
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
        
        return 0 

