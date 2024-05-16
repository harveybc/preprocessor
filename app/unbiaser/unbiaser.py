# -*- coding: utf-8 -*-
"""
This File contains the Unbiaser class.

"""

import argparse
import sys
import logging
import numpy as np
from preprocessor.preprocessor import Preprocessor
from itertools import zip_longest 
from joblib import dump, load

__author__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class Unbiaser(Preprocessor):
    """ The Unbiaser preprocessor class """

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
        parser = argparse.ArgumentParser(description="Unbias Moving Average: removes the moving average from a dataset.")
        parser.add_argument("--window_size", type=int, help="Window size for the moving average.", default=24)
        parser.add_argument("--no_config", help="Do not generate an output configuration file.", action="store_true", default=False)
        parser.add_argument("--headers", help="Handle headers in the dataset.", action="store_true", default=False)
        parser.add_argument("--detect_date", help="Auto-detect date column in the dataset.", action="store_true", default=False)

        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "no_config"):
            self.no_config = pargs.no_config
        else:
            self.no_config = False
        self.window_size = pargs.window_size

    def core(self):
        """ Core preprocessor task after starting the instance with the main method.
        Decide from the arguments, what method to call.

        Args:
        args (obj): command line parameters as objects
        """
        if hasattr(self, "input_config_file"):
            if self.input_config_file != None:
                self.load_from_config()
            else:
                self.unbias_ma()
        else:
            self.unbias_ma()

    def unbias_ma(self):
        """ Unbias the dataset using moving average. """
        self.output_ds = np.array([self.input_ds[i] - np.mean(self.input_ds[max(i - self.window_size + 1, 0):i+1]) for i in range(len(self.input_ds))])
        if hasattr(self, "no_config") and not self.no_config:
            config_data = {'window_size': self.window_size}
            dump(config_data, self.output_config_file)

    def load_from_config(self):
        """ Unbias the dataset from a config file. """
        config_data = load(self.input_config_file)
        self.window_size = config_data['window_size']
        self.output_ds = np.array([self.input_ds[i] - np.mean(self.input_ds[max(i - self.window_size + 1, 0):i+1]) for i in range(len(self.input_ds))])

    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",", fmt='%1.6f')

def run(args):
    """ Entry point for console_scripts """
    unbiaser = Unbiaser(None)
    unbiaser.main(args)

if __name__ == "__main__":
    run(sys.argv)