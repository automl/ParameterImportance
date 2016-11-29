from collections import OrderedDict
from sklearn.decomposition import PCA
from importance.evaluator.base_evaluator import AbstractEvaluator


class ForwardSelector(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'Forward Selection'

    def run(self) -> OrderedDict:
        params = self.cs.get_hyperparameters()
        used = []
        used.extend(range(len(params), len(self.model.types)))  # we don't want to evaluate the feature importance
        pca = PCA(n_components=min(7, len(self.types) - len(params)))
        self.scen.feature_array = pca.fit_transform(self.scen)

    def plot_result(self):
        pass
