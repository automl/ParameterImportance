import copy
import random
import time
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import mean_squared_error

from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class InfluenceModel(AbstractEvaluator):

    """
    Implementation of Influence Models
    """

    def __init__(self, scenario, cs, model, to_evaluate: int, rng, margin: float=None, threshold: float=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'InfluenceModel'
        self.logger = 'pimp.' + self.name
        self.model = LinearRegression()
        self.smac_model = model  # TODO Only use the SMAC model as container for X and y.
        self.all_X = copy.deepcopy(self.X)
        self.all_y = copy.deepcopy(self.y)

        # TODO better splitting
        self.testX = self.X[:int(len(self.X) * .1), :]
        self.X = self.X[int(len(self.X) * .1):, :]
        self.testy = self.y[:int(len(self.y) * .1)]
        self.y = self.y[int(len(self.y) * .1):]

        self.threshold = 0.000005 if threshold is None else threshold
        self.margin = 0.00000005 if margin is None else margin

    def _generate_random_trajectory(self, params, param_ids):
        """
        Simple method to shuffle around order in which parameters will be evaluated
        :param params:
        :param param_ids:
        :return:
        """
        tmp = list(range(len(params)))
        random.shuffle(tmp)
        return [params[i] for i in tmp], [param_ids[i] for i in tmp]

    def run(self) -> OrderedDict:
        """
        Implementation of the Stepwise feature selection as presented in
        "Performance-Influence Models for Highly Configurable Systems"
        :return: OrderedDict (param_name) -> (importance_value)
        """
        # Setup
        error = float('inf')
        lastError = float('inf')
        start_params = self.cs.get_hyperparameters()
        start_param_ids = list(range(len(start_params)))
        params = copy.deepcopy(start_params)
        param_ids = copy.deepcopy(start_param_ids)
        used = []  # Contains ids of which parametes will be used per round. Used for list-slicing
        # used.extend(range(len(params), len(self.smac_model.types)))  # we don't want to evaluate the feature importance

        # Beginning of the forward pass
        for num_evaluated in range(self.to_evaluate):
            best = None
            lastError = error
            params, param_ids = self._generate_random_trajectory(params, param_ids)

            counter = -1
            for idx, parameter in zip(param_ids, params):  # Along this trajectory sample the performance of parameters
                counter += 1
                self.logger.debug('Evaluating %s' % parameter.name)
                used.append(idx)  # similar to forward selection
                self.logger.debug('Used parameters: %s' % str(used))

                start = time.time()
                self.model.fit(self.X[:, used], self.y)  # refit the model every round
                round_error = mean_squared_error(self.testy, self.model.predict(self.testX[:, used]))
                self.logger.debug('Refitted LR (sec %.2f; error: %.4f)' % (time.time() - start, round_error))
                used.pop()
                if round_error < error:
                    error = round_error
                    best = parameter
                    best_idx = counter  # if we set it to idx we get the idx in start_params not which parameter to pop
                                        # from params and param_ids
            if error < self.threshold or (lastError - error) < self.margin:
                # If no imporvement is found this method automatically stops
                self.logger.debug('Stopping Forward pass')
                self.logger.debug('error < threshold: %s' % str(error < self.threshold))
                self.logger.debug('\delta error : %s' % str((lastError - error)))
                self.logger.debug('\delta error < margin: %s' % str((lastError - error) < self.margin))
                break
            else:
                if best is not None:
                    params.pop(best_idx)
                    used.append(param_ids.pop(best_idx))
                    self.logger.info('%s: %.4f (error)' % (best.name, error))
                    self.evaluated_parameter_importance[best.name] = self.model.coef_[-1]
                    best = None
                    best_idx = None
        remove = 0
        while remove >= 0:  # backward pass to determine if one parameter grew obsolete by adding others
            remove = -1
            for i in range(len(used)):
                tmp_used = used[:i] + used[i:]
                self.logger.debug('Used parameters: %s' % str(tmp_used))

                start = time.time()
                self.model.fit(self.X[:, used], self.y)  # refit the model every round
                round_error = mean_squared_error(self.testy, self.model.predict(self.testX[:, used]))
                # self.evaluated_parameter_importance[used[i]] = error
                self.logger.debug('Refitted LR (sec %.2f; error: %.4f)' % (time.time() - start, round_error))
                if round_error < error:  # if an obsolete parameter has been found.
                    error = round_error
                    remove = i  # remove it
                    del self.evaluated_parameter_importance[start_params[used[remove]].name]
                    self.logger.info('Removed %s' % start_params[used[remove]].name)
                    break
            if remove >= 0:
                used.pop(remove)
        self.model.fit(self.all_X[:, used], self.all_y)
        self.evaluated_parameter_importance = OrderedDict()
        coeffs = self.model.coef_.flatten()
        for idx, id in enumerate(used):
            self.evaluated_parameter_importance[start_params[id].name] = coeffs[idx]

        all_res = {'imp': self.evaluated_parameter_importance, 'order': list(self.evaluated_parameter_importance.keys())}
        return all_res

    def plot_result(self, name=None, show=True):
        # TODO find out what a sensible way of plotting this would be.
        # get sort index
        ids = [i[0] for i in sorted(enumerate(self.evaluated_parameter_importance.values()), key=lambda x: x[1])]

        fig, ax = plt.subplots()
        params = np.array(list(self.evaluated_parameter_importance.keys()))[ids]
        weights = np.array(list(self.evaluated_parameter_importance.values()))[ids]

        max_weight = max(weights) + abs(min(weights))
        tmp = np.arange(len(weights))
        bars = ax.bar(tmp, weights, color=self.area_color)

        for b, t in zip(enumerate(bars), ax.xaxis.get_ticklabels()):
            if (abs(weights[b[0]]) / max_weight) < 2*self.IMPORTANCE_THRESHOLD:
                b[1].set_color(self.unimportant_area_color)
                t.set_color((0.45, 0.45, 0.45))

        ax.set_ylabel('weights', **self.LABEL_FONT)
        ax.set_xticks(tmp)
        ax.set_xlim(-.5, len(tmp) - 0.5)
        ax.set_xticklabels(params, rotation=30, ha='right', **self.AXIS_FONT)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

        plt.ylim((-1*(abs(min(weights)) + .1*abs(min(weights))), abs(max(weights)) + .1*abs(max(weights))))

        plt.tight_layout()
        if name is not None:
            fig.savefig(name)
            if show:
                plt.show()
            self.logger.info('Saved plot as %s.png' % name)
        else:
            plt.show()
        plt.close('all')
