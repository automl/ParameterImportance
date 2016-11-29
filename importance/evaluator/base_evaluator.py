import abc
import logging
from collections import OrderedDict
from importance.epm import RandomForestWithInstances
from importance.configspace import ConfigurationSpace
from importance.utils import Scenario


class AbstractEvaluator(object):
    def __init__(self, scenario: Scenario,
                 cs: ConfigurationSpace,
                 model: RandomForestWithInstances,
                 to_evaluate: int, **kwargs):
        self.scenario = scenario
        self.cs = cs
        self.model = model  # SMAC model

        if 'X' in kwargs and 'y' in kwargs:
            self._train_model(kwargs['X'], kwargs['y'], **kwargs)
        if 'features' in kwargs:
            self.features = kwargs['features']
        else:
            self.features = self.model.instance_features

        self.X = self.model.X
        self.y = self.model.y
        self.types = self.model.types

        if to_evaluate <= 0:
            self.to_evaluate = len(self.cs.get_hyperparameters())
        else:
            self.to_evaluate = to_evaluate  # num of parameters to evaluate

        self.evaluated_parameter_importance = OrderedDict()
        self.name = 'Base'

    @abc.abstractclassmethod
    def run(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractclassmethod
    def plot_result(self):
        raise NotImplementedError

    def _train_model(self, X, y, **kwargs):
        self.model.train(X, y, **kwargs)

    def __str__(self):
        tmp = 'Parameter Importance Evaluation Method %s\n' % self.name
        tmp += '{:^15s}: {:<8s}\n'.format('Parameter', 'Value')
        for key in self.evaluated_parameter_importance:
            value = self.evaluated_parameter_importance[key]
            tmp += '{:>15s}: {:<3.4f}\n'.format(key, value)
        return tmp

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = logging.getLogger(value)

    def _refit_model(self, types, X, y):
        self.model = RandomForestWithInstances(types)
        self.model.rf.compute_oob_error = True
        self.model.train(X, y)
        return True
