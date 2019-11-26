from collections import OrderedDict
import pickle
import warnings
import copy

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import itertools as it
from tqdm import tqdm

from pimp.utils.bokeh_helpers import bokeh_boxplot, bokeh_line_uncertainty, save_and_show

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

try:
    from bokeh.io import show, output_file, save
    from bokeh.plotting import figure
    from bokeh.palettes import Inferno256
    from bokeh.layouts import Row
    from bokeh.models import ColumnDataSource, Span, CDSView, IndexFilter, BasicTickFormatter, \
        ContinuousColorMapper, LinearColorMapper, ColorBar, BasicTicker
    from bokeh.models import PrintfTickFormatter
    from bokeh.models.widgets import Tabs, Panel
    from bokeh.transform import transform
except ImportError as err:
    pass
    #self.logger.debug(err, exc_info=1)
    #self.logger.error("To use bokeh-plotting, you need to install bokeh (e.g. pip install bokeh)")

from pimp.evaluator.base_evaluator import AbstractEvaluator


__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class fANOVA(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, runhist: RunHistory, rng,
                 n_pairs=5, minimize=True, pairwise=True, preprocessed_X=None, preprocessed_y=None,
                 incumbents=None, **kwargs):
        """
        Handler to fANOVA module.

        Parameters
        ----------
        scenario: Scenario
            scenario with information about run_objective
        cs: ConfigSpace
            configuration space of scenario to be analyzed
        model: empirical performance model
            TODO
        to_evaluate: int
            number of parameters to be plotted
        runhist: RunHistory
            TODO
        rng: RandomNumberGenerator
            rng
        n_pairs: int
            how many (most important) parameters should be plotted for pairwise
            marginals
        minimize: boolean
            whether optimum is min or max
        pairwise: boolean
            plot pairwise marginals
        preprocessed_X/Y: data
            preprocessed data to be reused if model is already trained on data
            without instance_features
        incumbents: List[Configuration] or Configuration
            one or multiple incumbents to be marked in plots
        """
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'fANOVA'
        self.logger = 'pimp.' + self.name

        # Turn all Constants into Categoricals (fANOVA cannot handle Constants)
        self.cs_contained_constant = False
        # if any([isinstance(hp, Constant) for hp in self.cs.get_hyperparameters()]):
        #     self.logger.debug("Replacing configspace's hyperparameter Constants by one-value Categoricals.")
        #     new_hyperparameters = [CategoricalHyperparameter(hp.name, [hp.value]) if isinstance(hp, Constant)
        #                            else hp for hp in self.cs.get_hyperparameters()]
        #     self.cs = ConfigurationSpace()
        #     self.cs.add_hyperparameters(new_hyperparameters)
        #     self.cs_contained_constant = True

        # This way the instance features in X are ignored and a new forest is constructed
        if self.model.instance_features is None:
            self.logger.info('No marginalization necessary')
            if preprocessed_X is not None and preprocessed_y is not None:
                self._X = preprocessed_X
                self._y = preprocessed_y
            else:
                self.logger.info('Preprocessing X')
                self._X = copy.deepcopy(self.X)
                self._y = copy.deepcopy(self.y)
                for c_idx, config in enumerate(self.X):
                    # print("{}/{}".format(c_idx, len(self.X)))
                    for p_idx, param in enumerate(self.cs.get_hyperparameters()):
                        if not (isinstance(param, CategoricalHyperparameter) or
                                isinstance(param, Constant)):
                            # getting the parameters out of the hypercube setting as used in smac runhistory
                            self._X[c_idx][p_idx] = param._transform(self.X[c_idx][p_idx])
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
        self.evaluator = fanova_pyrfr(X=self._X, Y=self._y.flatten(), config_space=self.cs,
                                      seed=self.rng.randint(2**31-1), cutoffs=cutoffs)
        self.n_most_imp_pairs = n_pairs
        self.num_single = None
        self.pairwise = pairwise
        self.evaluated_parameter_importance_uncertainty = OrderedDict()
        self.incumbents = incumbents

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
        self._X = X_non_hyper
        self._y = y_prime
        self.logger.info('Size of training X after preprocessing: %s' % str(self.X.shape))
        self.logger.info('Size of training y after preprocessing: %s' % str(self.y.shape))
        self.logger.info('Finished Preprocessing')

    def _get_label(self, run_obj):
        if run_obj == 'runtime':
            label = 'runtime [sec]'
        elif run_obj == 'quality':
            label = 'cost'
        else:
            label = '%s' % self.scenario.run_obj
        return label

    def plot_result(self, name='fANOVA', show=True):
        if not os.path.exists(name):
            os.mkdir(name)

        self.plot_bokeh()

        vis = Visualizer(self.evaluator, self.cs, directory=name, y_label=self._get_label(self.scenario.run_obj))
        self.logger.info('Getting Marginals!')
        pbar = tqdm(range(self.to_evaluate), ascii=True, disable=not self.verbose)
        for i in pbar:
            plt.close('all')
            plt.clf()
            param = list(self.evaluated_parameter_importance.keys())[i]
            # Plot once in log, once linear
            for mode in [(True, '_log'), (False, '')]:
                outfile_name = os.path.join(name, param.replace(os.sep, "_") + mode[1] + ".png")
                # The try/except clause is only for back-compatibility with fanova <= 2.0.11
                try:
                    vis.plot_marginal(self.cs.get_idx_by_hyperparameter_name(param), log_scale=mode[0], show=False, incumbents=self.incumbents)
                except TypeError:
                    self.logger.debug("Plotting incumbents not supported by fanova < 2.0.12")
                    vis.plot_marginal(self.cs.get_idx_by_hyperparameter_name(param), log_scale=mode[0], show=False)
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

    def _bokeh_helper_pairwise_cat_cat(self, p1_name, p2_name, data):
        # Reshape into 1D array
        df = pd.DataFrame(data.stack(), columns=['zz']).reset_index()
        # Plot (this is really the bokeh-part...)
        source = ColumnDataSource(df)
        mapper = LinearColorMapper(palette=Inferno256, low=df['zz'].min(), high=df['zz'].max())
        p = figure(x_range=[str(c) for c in data.index], y_range=[str(c) for c in reversed(data.columns)],
                   toolbar_location=None, tools="")
        p.rect(x=p1_name, y=p2_name, width=1, height=1, source=source,
               fill_color=transform('zz', mapper),
               line_color=None)
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=20),
                             formatter=BasicTickFormatter(use_scientific=False))
        p.add_layout(color_bar, 'right')
        return p

    def plot_bokeh(self, plot_name=None, show_plot=True):
        vis = Visualizer(self.evaluator, self.cs, directory='.', y_label=self._get_label(self.scenario.run_obj))

        # Single marginals
        plots_single = []
        for param_name in self.evaluated_parameter_importance.keys():
            try:
                param = self.cs.get_hyperparameter(param_name)
            except KeyError as err:
                self.logger.debug(err, exc_info=1)
                continue

            incumbents = []
            #if not self.incumbents is None:
            #    incumbents = self.incumbents.copy() if isinstance(self.incumbents, list) else [self.incumbents]
            values = [c[param_name] for c in incumbents if param_name in c and c[param_name] is not None]

            if isinstance(param, (CategoricalHyperparameter, Constant)):
                labels = param.choices if isinstance(param, CategoricalHyperparameter) else str(param)
                mean, std = vis.generate_marginal(param_name)
                inc_indices = [labels.index(val) for val in values]

                p = bokeh_boxplot(labels, mean, std,
                                  x_label="runtime [sec]" if self.scenario.run_obj == "runtime" else "cost",
                                  y_label=param.name,
                                  runtime=self.scenario.run_obj=="runtime",
                                  inc_indices=inc_indices)

            else:
                mean, std, grid = vis.generate_marginal(param_name, 100)
                mean, std = np.asarray(mean), np.asarray(std)
                log_scale = param.log or (np.diff(grid).std() > 0.000001)
                inc_indices = [(np.abs(np.asarray(grid) - val)).argmin() for val in values]

                p = bokeh_line_uncertainty(grid, mean, std, log_scale,
                                           x_label="runtime [sec]" if self.scenario.run_obj == "runtime" else "cost",
                                           y_label=param.name,
                                           inc_indices=inc_indices)

            plots_single.append(Panel(child=Row(p), title=param_name))

        # Pairwise marginals
        most_important_ones = list(self.evaluated_parameter_importance.keys())[
                              :min(self.num_single, self.n_most_imp_pairs)]
        most_important_pairwise_marginals = vis.fanova.get_most_important_pairwise_marginals(params=most_important_ones)

        plots_pairwise = []
        for p1_name, p2_name in most_important_pairwise_marginals:
            p1, p2 = self.cs.get_hyperparameter(p1_name), self.cs.get_hyperparameter(p2_name)
            p1_idx = self.cs.get_idx_by_hyperparameter_name(p1_name)
            p2_idx = self.cs.get_idx_by_hyperparameter_name(p2_name)
            first_is_cat = isinstance(p1, CategoricalHyperparameter)
            second_is_cat = isinstance(p2, CategoricalHyperparameter)
            # There are essentially three cases / different plots:
            # First case: both categorical -> heatmap
            if first_is_cat or second_is_cat:
                choices, zz = vis.generate_pairwise_marginal((p1_idx, p2_idx), 20)
                # Working with pandas makes life easier
                data = pd.DataFrame(zz, index=choices[0], columns=choices[1])
                # Setting names for rows and columns and make categoricals strings
                data.index.name, data.columns.name = p1_name, p2_name
                data.index = data.index.astype(str) if first_is_cat else data.index
                data.columns = data.columns.astype(str) if second_is_cat else data.columns
                if first_is_cat and second_is_cat:
                    p = self._bokeh_helper_pairwise_cat_cat(p1_name, p2_name, data)
                else:
                    continue
                    raise NotImplementedError("fANOVA has to be fixed before it makes sense to work on this...")
                    # Only one of them is categorical -> create multi-line-plot
                    # Make distinction between categorical and noncategorical:
                    cat_name = p1_name if first_is_cat else p2_name
                    noncat_name = p1_name if second_is_cat else p2_name
                    # ASSUMING DATA IS NOT SWAPPED IN FANOVA (NOT SURE IF THIS HOLDS)
                    cat_choices = p1.choices if first_is_cat else p2.choices
                    # We want categorical values be represented by columns:
                    if not second_is_cat:
                        data = data.transpose()
                    # Find y_min and y_max BEFORE resetting index (otherwise index max obscure the query)
                    y_min, y_max = data.min().min(), data.max().max()
                    print(y_min)
                    # We want the index as a column (for plotting on x-axis)
                    print(data)
                    data = data.reset_index()
                    print(data)
                    source = ColumnDataSource(data)

                    print(data[noncat_name].min())
                    p = figure(x_range=(data[noncat_name].min(), data[noncat_name].max()),
                               y_range=(y_min, y_max),
                               toolbar_location=None, tools="")
                    for i, cat in enumerate(cat_choices):
                        p.line(x=noncat_name,
                               y=cat,
                               source=source)

            else:
                continue
                raise NotImplementedError("We might end up doing something like the 3DSurface example on bokeh")
                # Third case: both continous
            #outfile_name = os.path.join(self.directory, str(param_names).replace(" ", "_") + ".png")

            plots_pairwise.append(Panel(child=Row(p), title=p1_name+p2_name))

        print(len(plots_pairwise))
        print(len(plots_single))
        # Putting both together
        layout = Tabs(tabs=[*plots_pairwise,*plots_single])
        print(layout)
        show(layout)

        # Save and show...
        save_and_show(plot_name, show_plot, layout)

        return layout

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
