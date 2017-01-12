import time
import copy
import random
from collections import OrderedDict
from importance.evaluator.forward_selection import ForwardSelector

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class InfluenceModel(ForwardSelector):

    """
    Implementation of Influence Models
    """

    def __init__(self, scenario, cs, model, to_evaluate: int, margin: float=None, threshold: float=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'InfluenceModel'
        self.logger = self.name

        self.threshold = 0.0005 if threshold is None else threshold
        self.margin = 0.0005 if margin is None else margin

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
        start_params = self.cs.get_hyperparameters()
        start_param_ids = list(range(len(start_params)))
        params = copy.deepcopy(start_params)
        param_ids = copy.deepcopy(start_param_ids)
        used = []  # Contains ids of which parametes will be used per round. Used for list-slicing
        used.extend(range(len(params), len(self.model.types)))  # we don't want to evaluate the feature importance

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
                self._refit_model(self.types[used], self.X[:, used], self.y)  # refit the model every round
                oob = self.model.rf.out_of_bag_error()
                self.logger.debug('Refitted RF (sec %.2f; oob: %.4f)' % (time.time() - start, oob))
                used.pop()
                if oob < error:
                    error = oob
                    best = parameter
                    best_idx = counter  # if we set it to idx we get the idx in start_params not which parameter to pop
                                        # from params and param_ids
            if error < self.threshold or (lastError - error) < self.margin:
                # If no imporvement is found this method automatically stops
                break
            else:
                if best is not None:
                    params.pop(best_idx)
                    used.append(param_ids.pop(best_idx))
                    self.logger.info('%s: %.4f (OOB)' % (best.name, error))
                    self.evaluated_parameter_importance[best.name] = error
                    best = None
                    best_idx = None
        remove = 0
        while remove >= 0:  # backward pass to determine if one parameter grew obsolete by adding others
            remove = -1
            for i in range(len(used)):
                tmp_used = used[:i] + used[i:]
                self.logger.debug('Used parameters: %s' % str(tmp_used))

                start = time.time()
                self._refit_model(self.types[used], self.X[:, used], self.y)  # refit the model every round
                oob = self.model.rf.out_of_bag_error()
                # self.evaluated_parameter_importance[used[i]] = error
                self.logger.debug('Refitted RF (sec %.2f; oob: %.4f)' % (time.time() - start, oob))
                if oob < error:  # if an obsolete parameter has been found.
                    error = oob
                    remove = i  # remove it
                    del self.evaluated_parameter_importance[start_param_ids[used[remove]].name]
                    self.logger.info('Removed %s' % start_params[used[remove]].name)
                    break
            if remove >= 0:
                used.pop(remove)

        return self.evaluated_parameter_importance

    def plot_result(self, name=None):
        """
        Calls Forwad-Selections plotting function as it can be plotted the same way
        :param name: File name
        """
        super().plot_result(name)
