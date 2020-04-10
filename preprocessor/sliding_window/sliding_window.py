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
from collections import deque

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
        """ Parse command line parameters additional to the preprocessor class ones

        Args:
            args ([str]): command line parameters as list of strings

        Returns:
            :obj:`argparse.Namespace`: command line parameters namespace
        """
        parser = argparse.ArgumentParser(
            description="SlidingWindow: performs the sliding window technique on the input dataset."
        )
        parser.add_argument("--window_size",
            help="Size of the window to be use for the sliding window technique. Default 30",
            type=int,
            default=30
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        if hasattr(pargs, "window_size"):
            self.window_size = pargs.window_size
        else:
            self.window_size = 30
        self.assign_arguments(pargs)
        
    def core(self):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what method to call.

        Args:
        args (obj): command line parameters as objects
        """
        self.sl_window()
        
    def sl_window(self):
        """ Perform sliding window technique on the input the dataset. """
        # initialize output dataset
        out_ds = []
        window = deque(self.input_ds[0:self.window_size-1, :], self.window_size)
        # initialize window and window_future para cada tick desde 0 hasta window_size-1
        for i in range(1, self.window_size+1):
            tick_data = self.input_ds[i, :].copy()
            # fills the training window with past data
            window.appendleft(tick_data.copy())
        # para cada tick desde window_size hasta num_ticks - 1
        for i in range(self.window_size, self.num_ticks-self.window_size):
            tick_data = self.input_ds[i, :].copy()
            # fills the training window with past data
            window.appendleft(tick_data.copy())
            # expande usando los window tick anteriores (traspuesta de la columna del feature en la matriz window)
            for it,v in enumerate(tick_data):
                w_count = 0
                for w in window:
                    if (w_count == 0) and (it==0):
                        window_column_t = [w[it]]
                    else:
                        window_column_t = np.concatenate((window_column_t, [w[it]]))
                    w_count = w_count + 1
                tick_data_r = window_column_t.copy()
            out_ds.append(tick_data_r)
            if i % 100 == 0.0:
                progress = i*100/num_ticks
                sys.stdout.write("Tick: %d/%d Progress: %d%%   \r" % (i, num_ticks, progress) )
                sys.stdout.flush()
        self.output_ds = np.array(out_ds)
                    
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
