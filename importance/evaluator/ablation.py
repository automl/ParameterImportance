from collections import OrderedDict

from importance.evaluator.base_evaluator import AbstractEvaluator


class Ablation(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'Ablation'

    def run(self) -> OrderedDict:
        pass

    def plot_result(self):
        pass
