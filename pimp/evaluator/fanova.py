from collections import OrderedDict
from pimp.evaluator.base_evaluator import AbstractEvaluator
# from importance.evaluator import fanova_pyrfr, Visualizer

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"

# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO This is a Placeholder
# TODO Currently fANOVA is not supported

class fANOVA(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'fANOVA'
        self.logger = self.name
        # This way the instance features in X are ignored and a new forest is constructed
        # TODO figure out if an already trained forest from the model can be reused!
        self.evaluator = fanova_pyrfr(X=model.X[:, :model.X.shape[1] - len(model.instance_features[0])],
                                      Y=model.y, cs=cs)

    def plot_result(self, name=None):
        vis = Visualizer(self.evaluator, self.cs)
        vis.create_all_plots('.')

    def run(self) -> OrderedDict:
        raise NotImplementedError
        # return self.evaluator.get_marginal()
        # fanova.get_marginal list dimension list / position of parameters in configspace to analyze

        # for parameter in self.to_evaluate:
        #     get_idx in ordered_dict_of_config_space
        #     imp_val_of_param[parameter] = fanova.get_marginal(dim_list=[idx])
