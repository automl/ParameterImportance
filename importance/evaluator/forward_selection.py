from collections import OrderedDict

from importance.evaluator.base_evaluator import AbstractEvaluator


class ForwardSelector(AbstractEvaluator):

    def __init__(self, cs, model, to_evaluate: list, **kwargs):
        super().__init__(cs, model, to_evaluate, **kwargs)
        self.name = 'Forward Selection'

    def run(self) -> OrderedDict:
        pass

    def plot_result(self):
        pass
