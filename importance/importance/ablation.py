from importance.importance.base_evaluator import AbstractEvaluator
from collections import OrderedDict


class Ablation(AbstractEvaluator):

    def __init__(self):
        self.name = 'Ablation'

    def run(self) -> OrderedDict:
        pass

    def plot_result(self):
        pass
