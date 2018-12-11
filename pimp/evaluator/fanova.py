from collections import OrderedDict
import pickle
import warnings

import os
import numpy as np
import matplotlib as mpl
import itertools as it
from tqdm import tqdm
mpl.use('Agg')
from matplotlib import pyplot as plt

from smac.runhistory.runhistory import RunHistory
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.util import impute_inactive_values
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant

try:
    from fanova import fANOVA as fanova_pyrfr
    from fanova.visualizer import Visualizer
except ImportError:
    warnings.simplefilter('always', ImportWarning)
    warnings.warn('\n{0}\n{0}{1}{0}\n{0}'.format('!'*120,
                                     '\nfANOVA is not installed in your environment. To install it please run '
                                     '"git+http://github.com/automl/fanova.git@master"\n'))

from pimp.evaluator.base_evaluator import AbstractEvaluator


__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class fANOVA(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, runhist: RunHistory, rng,
                 n_pairs=5, minimize=True, pairwise=True, preprocessed_X=None, preprocessed_y=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'fANOVA'
        self.logger = 'pimp.' + self.name

        # Turn all Constants into Categoricals (fANOVA cannot handle Constants)
        self.cs_contained_constant = False
        if any([isinstance(hp, Constant) for hp in self.cs.get_hyperparameters()]):
            self.logger.debug("Replacing configspace's hyperparameter Constants by one-value Categoricals.")
            new_hyperparameters = [CategoricalHyperparameter(hp.name, [hp.value]) if isinstance(hp, Constant)
                                   else hp for hp in self.cs.get_hyperparameters()]
            self.cs = ConfigurationSpace()
            self.cs.add_hyperparameters(new_hyperparameters)
            self.cs_contained_constant = True

        # This way the instance features in X are ignored and a new forest is constructed
        if self.model.instance_features is None:
            self.logger.info('No preprocessing necessary')
            if preprocessed_X is not None and preprocessed_y is not None:
                self.X = preprocessed_X
                self.y = preprocessed_y
            else:
                self._preprocess(runhist)
        else:
            self._preprocess(runhist)
        cutoffs = (-np.inf, np.inf)
        if minimize:
            cutoffs = (-np.inf, self.model.predict_marginalized_over_instances(
                np.array([impute_inactive_values(self.cs.get_default_configuration()).get_array()]))[0].flatten()[0]
                       )
        elif minimize is False:
            cutoffs = (self.model.predict_marginalized_over_instances(
                np.array([impute_inactive_values( self.cs.get_default_configuration()).get_array()]))[0].flatten()[0],
                       np.inf)
        self.evaluator = fanova_pyrfr(X=self.X, Y=self.y.flatten(), config_space=self.cs,
                                      seed=self.rng.randint(2**31-1), cutoffs=cutoffs)
        self.n_most_imp_pairs = n_pairs
        self.num_single = None
        self.pairwise = pairwise
        self.evaluated_parameter_importance_uncertainty = OrderedDict()

    def _preprocess(self, runhistory):
        """
        Method to marginalize over instances such that fANOVA can determine the parameter importance without
        having to deal with instance features.
        :param runhistory: RunHistory that knows all configurations that were run. For all these configurations
                           we have to marginalize away the instance features with which fANOVA will make it's
                           predictions
        """
        self.logger.info('PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING')
        self.logger.info('Marginalizing away all instances!')
        configs = runhistory.get_all_configs()
        if self.cs_contained_constant:
            configs = [Configuration(self.cs, vector=c.get_array()) for c in configs]
        X_non_hyper, X_prime = [], []
        for config in configs:
            config = impute_inactive_values(config).get_array()
            X_prime.append(config)
            X_non_hyper.append(config)
            for idx, param in enumerate(self.cs.get_hyperparameters()):
                if not (isinstance(param, CategoricalHyperparameter) or
                        isinstance(param, Constant)):
                    X_non_hyper[-1][idx] = param._transform(X_non_hyper[-1][idx])
        X_non_hyper = np.array(X_non_hyper)
        X_prime = np.array(X_prime)
        y_prime = np.array(self.model.predict_marginalized_over_instances(X_prime)[0])
        self.X = X_non_hyper
        self.y = y_prime
        self.logger.info('Size of training X after preprocessing: %s' % str(self.X.shape))
        self.logger.info('Size of training y after preprocessing: %s' % str(self.y.shape))
        self.logger.info('Finished Preprocessing')

    def plot_result(self, name='fANOVA', show=True):
        if not os.path.exists(name):
            os.mkdir(name)

        if self.scenario.run_obj == 'runtime':
            label = 'runtime [sec]'
        elif self.scenario.run_obj == 'quality':
            label = 'cost'
        else:
            label = '%s' % self.scenario.run_obj
        vis = Visualizer(self.evaluator, self.cs, directory=name, y_label=label)
        self.logger.info('Getting Marginals!')
        pbar = tqdm(range(self.to_evaluate), ascii=True, disable=not self.verbose)
        for i in pbar:
            plt.close('all')
            plt.clf()
            param = list(self.evaluated_parameter_importance.keys())[i]
            outfile_name = os.path.join(name, param.replace(os.sep, "_") + ".png")
            vis.plot_marginal(self.cs.get_idx_by_hyperparameter_name(param), log_scale=False, show=False)
            fig = plt.gcf()
            fig.savefig(outfile_name)
            plt.close('all')
            plt.clf()
            outfile_name = os.path.join(name, param.replace(os.sep, "_") + "_log.png")
            vis.plot_marginal(self.cs.get_idx_by_hyperparameter_name(param), log_scale=True, show=False)
            fig = plt.gcf()
            fig.savefig(outfile_name)
            plt.close('all')
            plt.clf()
            if show:
                plt.show()
            pbar.set_description('Creating fANOVA plot: {: <.30s}'.format(outfile_name.split(os.path.sep)[-1]))
        if self.pairwise:
            self.logger.info('Plotting Pairwise-Marginals!')
            most_important_ones = list(self.evaluated_parameter_importance.keys())[
                                  :min(self.num_single, self.n_most_imp_pairs)]
            try:
                vis.create_most_important_pairwise_marginal_plots(most_important_ones)
            except TypeError:
                self.logger.warning('Could not create pairwise plots!')
            plt.close('all')

    def run(self) -> OrderedDict:
        try:
            params = self.cs.get_hyperparameters()

            tmp_res = []
            for idx, param in enumerate(params):
                imp = self.evaluator.quantify_importance([idx])[(idx, )]
                self.logger.debug('{:>02d} {:<30s}: {:>02.4f}' .format(
                    idx, param.name, imp['total importance']))
                tmp_res.append(imp)

            tmp_res_sort_keys = [i[0] for i in sorted(enumerate(tmp_res), key=lambda x:x[1]['total importance'], reverse=True)]
            self.logger.debug(tmp_res_sort_keys)
            count = 0
            for idx in tmp_res_sort_keys:
                if count >= self.to_evaluate:
                    break
                self.logger.info('{:>02d} {:<30s}: {:>02.4f}'.format(idx, params[idx].name, tmp_res[idx]['total importance']))
                self.evaluated_parameter_importance[params[idx].name] = tmp_res[idx]['total importance']
                try:
                    self.evaluated_parameter_importance_uncertainty[params[idx].name] = tmp_res[idx]['total std']
                except KeyError as e:
                    self.logger.debug("std not available yet for this fanova version")
                count += 1
            self.num_single = len(list(self.evaluated_parameter_importance.keys()))
            if self.pairwise:
                self.logger.info(
                    'Computing most important pairwise marginals using at most'
                    ' the %d most important ones.' % min(self.n_most_imp_pairs, self.num_single))
                pairs = [x for x in it.combinations(list(self.evaluated_parameter_importance.keys())[:self.n_most_imp_pairs], 2)]
                for pair in pairs:
                    imp = self.evaluator.quantify_importance(pair)[pair]
                    try:
                        mean, std = imp['individual importance'], imp['individual std']
                        self.evaluated_parameter_importance_uncertainty[str(list(pair))] = std
                    except KeyError as e:
                        self.logger.debug("std not available yet for this fanova version")
                        mean= imp['individual importance']
                    self.evaluated_parameter_importance[str(list(pair))] = mean
                    a, b = pair
                    if len(a) > 13:
                        a = str(a)[:5] + '...' + str(a)[-5:]
                    if len(b) > 13:
                        b = str(b)[:5] + '...' + str(b)[-5:]
                    self.logger.info('{:>02d} {:<30s}: {:>02.4f}'.format(-1, a + ' <> ' + b, mean))
            all_res = {'imp': self.evaluated_parameter_importance,
                       'order': list(self.evaluated_parameter_importance.keys())}
            return all_res
        except ZeroDivisionError:
            with open('fANOVA_crash_data.pkl', 'wb') as fh:
                pickle.dump([self.X, self.y, self.cs], fh)
            raise Exception('fANOVA crashed with a "float division by zero" error. Dumping the data to disk')
