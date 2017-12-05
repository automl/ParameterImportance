import time
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class ForwardSelector(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, rng, feature_imp: bool=False, **kwargs):
        """
        Constructor
        :parameter:
        scenario
            SMAC scenario object
        cs
            ConfigurationSpace object
        model
            SMACs EPM (RF)
        to_evaluate
            int. Indicates for how many parameters the Importance values have to be computed
        """
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'Forward Selection'
        self.logger = self.name
        self.feature_importance = feature_imp
        self.logger.info('%slyzing feature importance' % ('A' if feature_imp else 'Not a'))

    def run(self) -> OrderedDict:
        """
        Implementation of the forward selection loop.
        Uses SMACs EPM (RF) wrt the configuration space to minimize the OOB error.
        :return:
        OrderedDict
            dict_keys (first key -> most important) -> OOB error
        """
        params = self.cs.get_hyperparameters()
        param_ids = list(range(len(params)))
        feature_ids = list(range(len(params), len(self.types)))
        used = []
        used_bounds = []
        if self.feature_importance:
            used.extend(range(0, len(self.bounds)))
            used_bounds.extend(range(0, len(self.bounds)))
            names = list(map(lambda x: 'Feat #' + str(len(feature_ids) + x - len(self.types)), feature_ids))
            ids = feature_ids
            if self.to_evaluate > len(feature_ids):
                self.to_evaluate = len(feature_ids)
        else:
            used.extend(range(len(params), len(self.model.types)))  # we don't want to evaluate the feature importance
            names = params
            ids = param_ids

        for _ in range(self.to_evaluate):  # Main Loop
            errors = []
            for idx, name in zip(ids, names):
                self.logger.debug('Evaluating %s' % name)
                used.append(idx)
                if not self.feature_importance:
                    used_bounds.append(idx)
                self.logger.debug('Used parameters: %s' % str(used))
                self.logger.debug('Used bounds of parameters: %s' % str(used_bounds))

                start = time.time()
                self._refit_model(self.types[sorted(used)], self.bounds[sorted(used_bounds)],
                                  self.X[:, sorted(used)], self.y)  # refit the model every round
                errors.append(np.sqrt(
                    np.mean((self.model.predict(self.X[:, sorted(used)])[0].flatten() - self.y) ** 2)))
                used.pop()
                if not self.feature_importance:
                    used_bounds.pop()
                self.logger.debug('Refitted RF (sec %.2f; error: %.4f)' % (time.time() - start, errors[-1]))

            best_idx = np.argmin(errors)
            lowest_error = errors[best_idx]
            best_parameter = names.pop(best_idx)
            used.append(ids.pop(best_idx))
            if not self.feature_importance:
                used_bounds.append(used[-1])

            if self.feature_importance:
                self.logger.info('%s: %.4f' % (best_parameter, lowest_error))
                self.evaluated_parameter_importance[best_parameter] = lowest_error
            else:
                self.logger.info('%s: %.4f' % (best_parameter.name, lowest_error))
                self.evaluated_parameter_importance[best_parameter.name] = lowest_error
        all_res = {'imp': self.evaluated_parameter_importance, 'order': list(self.evaluated_parameter_importance.keys())}
        return all_res

    def _plot_result(self, name, bar=True, show=True):
        """
            plot oob score as bar charts
            Parameters
            ----------
            name
                file name to save plot
        """

        fig, ax = plt.subplots()
        params = list(self.evaluated_parameter_importance.keys())
        errors = list(self.evaluated_parameter_importance.values())
        max_to_plot = min(len(errors), self.MAX_PARAMS_TO_PLOT)

        ind = np.arange(len(errors))
        if bar:
            ax.bar(ind, errors, color=self.area_color)
        else:
            ax.plot(ind, errors, **self.LINE_FONT)

        ax.set_ylabel('error', **self.LABEL_FONT)
        if bar:
            ax.set_xticks(ind)
            ax.set_xlim(-.5, max_to_plot - 0.5)
        else:
            ax.set_xticks(ind)
            ax.set_xlim(0, max_to_plot - 1)
        ax.set_xticklabels(params, rotation=30, ha='right', **self.AXIS_FONT)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

        plt.tight_layout()
        if name is not None:
            fig.savefig(name)
            if show:
                plt.show()
        else:
            plt.show()

    def plot_result(self, name=None, show=True):
        self._plot_result(name + '-barplot.png', True, show)
        self._plot_result(name + '-chng.png', False, show)
        plt.close('all')
        self.logger.info('Saved plot as %s-[barplot|chng].png' % name)
