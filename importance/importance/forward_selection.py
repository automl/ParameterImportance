from importance.importance.base_evaluator import AbstractEvaluator
from collections import OrderedDict


class ForwardSelector(AbstractEvaluator):

    def __init__(self):
        self.name = 'Forward Selection'

    def run(self) -> OrderedDict:
        pass

    def plot_result(self):
        pass
