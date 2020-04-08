# -*- coding: utf-8 -*-
"""
This File contains the SlidingWindow class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        sliding_window = sliding_window.__main__:main

Then run `python setup.py install` which will install the command `sliding_window`
inside your current environment.

"""

import argparse
import sys
import logging
import numpy as np
from sklearn import preprocessing
from preprocessor.preprocessor import Preprocessor
from itertools import zip_longest 
from joblib import dump, load

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class SlidingWindow(Preprocessor):
    """ The SlidingWindow preprocessor class """

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
        parser = argparse.ArgumentParser(
            description="SlidingWindow: performs the sliding window technique on the input dataset."
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        
    def core(self):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what method to call.

        Args:
        args (obj): command line parameters as objects
        """
        self.window()
        
    def window(self):
        """ Perform sliding window on the input the dataset. """
        pt = preprocessing.StandardScaler()
        pt.fit(self.input_ds) 
        self.output_ds = pt.transform(self.input_ds) 
        dump(pt, self.output_config_file)
        
    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",", fmt='%1.6f')

def run(args):
    """ Entry point for console_scripts """
    sliding_window = SlidingWindow(None)
    sliding_window.main(args)


if __name__ == "__main__":
    run(sys.argv)
