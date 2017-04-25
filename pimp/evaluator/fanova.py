from collections import OrderedDict

import numpy as np
from fanova.fanova import fANOVA as fanova_pyrfr
from fanova.visualizer import Visualizer

from pimp.evaluator.base_evaluator import AbstractEvaluator


__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class fANOVA(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'fANOVA'
        self.logger = self.name
        # This way the instance features in X are ignored and a new forest is constructed
        self._preprocess(self.X)
        self.evaluator = fanova_pyrfr(X=self.X, Y=self.y.flatten(), config_space=cs, config_on_hypercube=True)

    def _preprocess(self, X):
        """
        Method to marginalize over instances such that fANOVA can determine the parameter importance without
        having to deal with instance features.
        :param X: ndarray [n_samples, (m_parameters + m_instance_features)]
        :param y: ndarray [n_samples, ] of performance values for corresponding entries in X
        :return: X', y' such that configurations in X are marginalized over all instances resulting in y'
        """
        self.logger.info('PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING')
        y_prime = []
        X_prime = []
        dupes = 0
        tmp = []
        for row_x in X[:, :X.shape[1] - len(self.model.instance_features[0])]:
            if row_x.tolist() in tmp:  # needed to filter out duplicats
                dupes += 1
            else:
                X_prime.append(row_x)
                tmp.append(row_x.tolist())
                y_prime.append(self.model.predict_marginalized_over_instances(np.array([row_x]))[0])

        y_prime = np.array(y_prime)
        X_prime = np.array(X_prime)
        self.logger.debug('Duplicate Configurations in X found: %d' % dupes)
        self.logger.debug('Remaining Configurations in X: %d' % X_prime.shape[0])
        self.X = X_prime
        self.y = y_prime

        self.logger.info('Finished Preprocessing')

    def plot_result(self, name=None):
        vis = Visualizer(self.evaluator, self.cs)
        vis.create_all_plots('.')

    def run(self) -> OrderedDict:
        print(self.evaluator.quantify_importance(1))
