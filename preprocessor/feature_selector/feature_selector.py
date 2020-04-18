# -*- coding: utf-8 -*-
"""
This File contains the FeatureSelector class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        feature_selector = feature_selector.__main__:main

Then run `python setup.py install` which will install the command `feature_selector`
inside your current environment.

"""

import argparse
import sys
import logging
import numpy as np
from sklearn.feature_selection import SelectPercentile
from preprocessor.preprocessor import Preprocessor
from itertools import zip_longest 
from joblib import dump, load

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)
 
def score_func_regression(X,Y):
    """ Used to score the features for feature selection, for regression. To be used in the fFeatureSeclector.feature_selection() method. """
    import sklearn
    return sklearn.feature_selection.mutual_info_regression(X,Y) 

def score_func_classification(X,Y):
    """ Used to score the features for feature selection, for regression. To be used in the fFeatureSeclector.feature_selection() method. """
    import sklearn
    return sklearn.feature_selection.mutual_info_classif(X,Y) 

class FeatureSelector(Preprocessor):
    """ The FeatureSelector preprocessor class """

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
        _logger.debug("Parsing command-line arguments.")
        parser = argparse.ArgumentParser(
            description="Dataset FeatureSelector: select the best scoring features according to a mutual information scoring algorithm."
        )
        parser.add_argument("--training_file",
            help="filename of the training dataset"
        )
        parser.add_argument("--percentile",
            help="percentile of features to keep",
            type=int,
            default=10
        )
        parser.add_argument("--classification",
            help="Uses a classification training signal instead of regression that is the default if this parameter is not set.",
            action="store_true",
            default=False
        )
        parser.add_argument("--no_config",
            help="Do not generate an output configuration file.",
            action="store_true",
            default=False
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "no_config"):
            self.no_config = pargs.no_config
        else:
            self.no_config = False
        if hasattr(pargs, "percentile"):
            self.percentile = pargs.percentile
        else:
            self.percentile = 20
        if hasattr(pargs, "classification"):
            self.classification = True
        else:
            self.classification = False
        if hasattr(pargs, "training_file"):
            self.training_file = pargs.training_file
        else:
            print("Error: No training file parameter provided. Use option -h to show help.")
            sys.exit()

    def core(self):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what method to call.

        Args:
        args (obj): command line parameters as objects
        """
        _logger.debug("Performing core module task.")       
        if hasattr(self, "input_config_file"):
            if self.input_config_file != None:
                _logger.debug("Loading configuration file.")
                self.load_from_config()
            else:
                _logger.debug("Performing feature_selection() method.")
                self.feature_selection()
                _logger.debug("End feature_selection() method.")
        else:
            _logger.debug("Performing feature_selection() method..")
            self.feature_selection()  
            _logger.debug("End feature_selection() method..")

    def feature_selection(self):
        """ Process the dataset. """
        # loads the training file
        self.training_ds = np.genfromtxt(self.training_file, delimiter=",")
        # Initialize feature selector    
        if not(hasattr(self, "classification")):
            self.classification = False
        if not(hasattr(self, "percentile")):
            self.percentile = 20
        if self.classification:
            featureSelector = SelectPercentile(score_func = score_func_classification, percentile = self.percentile)
        else:
            featureSelector = SelectPercentile(score_func= score_func_regression, percentile = self.percentile)
        # fit feature selector using the training signal
        featureSelector.fit(self.input_ds, self.training_ds)
        # applies feature selection mask to the input dataset
        mask = featureSelector.get_support()
        self.output_ds = self.input_ds[:, mask]    
        # saves configuration file
        if not(hasattr(self, "no_config")):
            self.no_config = False
        if not(self.no_config):
            np.savetxt(self.output_config_file, mask, delimiter=",")

    def load_from_config(self):
        """ Process the dataset from a config file. """
        _logger.debug("Loading configuration from input_config_file = "+ self.input_config_file)
        mask = np.genfromtxt(self.input_config_file, delimiter=",")
        self.output_ds = self.input_ds[:, mask]
        
    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        _logger.debug("Storing output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",", fmt='%1.6f')

def run(args):
    """ Entry point for console_scripts """
    feature_selector = FeatureSelector(None)
    feature_selector.main(args)


if __name__ == "__main__":
    run(sys.argv)
