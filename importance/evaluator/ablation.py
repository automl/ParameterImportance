from collections import OrderedDict
import numpy as np
from importance.evaluator.base_evaluator import AbstractEvaluator
import copy


class Ablation(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, incumbent=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'Ablation'
        self.logger = self.name

        self.target = incumbent
        self.source = self.cs.get_default_configuration()
        self.delta = self._diff_in_source_and_target()

    def _diff_in_source_and_target(self):
        delta = []
        for parameter in self.source:
            tmp = ' not'
            if self.source[parameter] != self.target[parameter]:
                tmp = ''
                delta.append([parameter])

            self.logger.debug('%s was%s modified from source to target' % (parameter, tmp))
        return delta

    def run(self) -> OrderedDict:

        modifiable_config = copy.deepcopy(self.source.get_dictionary())
        modified_so_far = []

        while len(self.delta) > 0:
            for param_tuple in modified_so_far:  # necessary due to combined flips
                for parameter in param_tuple:
                    modifiable_config[parameter] = self.target[parameter]

            round_performances = []
            for param_tuple in self.delta:  # necessary due to combined flips
                for parameter in param_tuple:
                    modifiable_config[parameter] = self.target[parameter]
                    for children in self.cs.get_children_of(parameter):
                        print(children)
            self.delta.pop()


    def plot_result(self):
        pass
