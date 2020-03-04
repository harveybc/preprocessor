# -*- coding: utf-8 -*-
"""
This File contains the Standardizer class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        Standardizer = Standardizer.Standardizer

Then run `python setup.py install` which will install the command `Standardizer`
inside your current environment.

TODO: VERIFICAR

"""

import argparse
import sys
import logging
import numpy as np
from preprocessor.preprocessor import Preprocessor

# from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class Standardizer(Preprocessor):
    """ The Standardizer preprocessor class """

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
            description="Standardizer: standarize a dataset, exports or imports a configuration file for standarization."
        )
        parser = self.parse_cmd(parser)
        return parser.parse_args(args)

    def core(self, args):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """
        # Standarize dataset

    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        np.savetxt(self.input_file, self.output_ds, delimiter=",")

def run():
    """ Entry point for console_scripts """
    data_trimmer = DataTrimmer(None)
    data_trimmer.main(sys.argv[1:])


if __name__ == "__main__":
    run()
