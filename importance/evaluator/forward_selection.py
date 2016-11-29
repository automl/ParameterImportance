from collections import OrderedDict
from sklearn.decomposition import PCA
from importance.evaluator.base_evaluator import AbstractEvaluator
import time
import numpy as np


class ForwardSelector(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, **kwargs):
        '''
        Constructor
        :parameter:
        scenario
            SMAC scenario object
        cs
            ConfigurationSpace object
        model
            SMACs EPM (RF)
        to_evaluate
            int. Indicates for how many parameters the Importance values have to be computed
        '''
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'Forward Selection'
        self.logger = self.name

    def run(self) -> OrderedDict:
        '''
        Implementation of the forward selection loop.
        Uses SMACs EPM (RF) wrt the configuration space to minimize the OOB error.
        :return:
        OrderedDict
            dcit_keys (first key -> most important) -> OOB error
        '''
        params = self.cs.get_hyperparameters()
        param_ids = list(range(len(params)))
        used = []
        used.extend(range(len(params), len(self.model.types)))  # we don't want to evaluate the feature importance
        num_feats = len(self.types) - len(params)
        if num_feats > 0:
            pca = PCA(n_components=min(7, len(self.types) - len(params)))
            self.scenario.feature_array = pca.fit_transform(self.scenario.feature_array)

        for _ in range(self.to_evaluate):
            errors = []
            for idx, parameter in zip(param_ids, params):
                self.logger.debug('Evaluating %s' % parameter)
                used.append(idx)
                self.logger.debug('Used parameters: %s' % str(used))

                start = time.time()
                self._refit_model(self.types[used], self.X[:, used], self.y)
                errors.append(self.model.rf.out_of_bag_error())
                used.pop()
                self.logger.debug('Refitted RF (sec %.2f; oob: %.4f)' % (time.time() - start, errors[-1]))

            best_idx = np.argmin(errors)
            lowest_error = errors[best_idx]
            best_parameter = params.pop(best_idx)
            used.append(param_ids.pop(best_idx))

            self.logger.info('%s: %.4f (OOB)' % (best_parameter.name, lowest_error))
            self.evaluated_parameter_importance[best_parameter.name] = lowest_error
        return self.evaluated_parameter_importance

    def plot_result(self):
        pass
