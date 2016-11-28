import abc
from collections import OrderedDict

class AbstractEvaluator(object):
    def __init__(self, cs, model, to_evaluate):
        self.cs = cs
        self.model = model  #Should be SMAC model
        self.to_evaluate = to_evaluate  # list of parameter_names!!!

    @abc.abstractclassmethod
    def run(self) -> OrderedDict:
        raise NotImplementedError

