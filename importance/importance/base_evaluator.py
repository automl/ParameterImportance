import abc
from collections import OrderedDict
from importance.epm import RandomForestWithInstances
from importance.configspace import ConfigurationSpace


class AbstractEvaluator(object):
    def __init__(self, cs: ConfigurationSpace,
                 model: RandomForestWithInstances,
                 to_evaluate: list, **kwargs):
        self.cs = cs
        self.to_evaluate = to_evaluate  # list of parameter_names!!!
        self.model = model  #Should be SMAC model

        if 'X' in kwargs and 'y' in kwargs:
            self._train_model(kwargs['X'], kwargs['y'], **kwargs)
        if 'features' in kwargs:
            self.features = kwargs['features']
        else:
            self.features = self.model.instance_features

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
        return self.name
