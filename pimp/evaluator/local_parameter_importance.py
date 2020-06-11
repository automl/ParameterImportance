import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from bokeh.models import Row, Panel, Tabs
from tqdm import tqdm

from pimp.utils.bokeh_helpers import bokeh_boxplot, bokeh_line_uncertainty, save_and_show

tqdm.monitor_interval = 0
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from pimp.configspace import change_hp_value, Configuration, ForbiddenValueError,\
    impute_inactive_values, CategoricalHyperparameter, NumericalHyperparameter, check_forbidden
from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class LPI(AbstractEvaluator):

    """
    Implementation of Ablation via surrogates
    """

    def __init__(self, scenario, cs, model, to_evaluate: int, incumbent=None, continous_neighbors=500,
                 old_sampling=False, show_query_points=False, quant_var=True, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'LPI'
        self.logger = 'pimp.' + self.name
        self.incumbent = incumbent
        self.incumbent_dict = self.incumbent.get_dictionary()
        self._continous_param_neighbor_samples = continous_neighbors
        self.show_query_points = show_query_points
        self.old_sampling = old_sampling
        self.neighborhood_dict = None
        self.performance_dict = {}
        self._sampled_neighbors = 0
        self.variance_dict = {}
        self.quantify_importance_via_variance = quant_var
        self.evaluated_parameter_importance_uncertainty = OrderedDict()

    def _old_sampling_of_one_exchange_neighborhood(self, param, array, index):
        neighbourhood = []
        number_of_sampled_neighbors = 0
        iteration = 0
        checked_neighbors = []
        checked_neighbors_non_unit_cube = []
        while True:
            hp = self.incumbent.configuration_space.get_hyperparameter(param)
            num_neighbors = hp.get_num_neighbors(self.incumbent.get(param))
            self.logger.debug('\t' + str(num_neighbors))

            # Obtain neigbors differently for different possible numbers of
            # neighbors
            if num_neighbors == 0:
                self.logger.debug('\tNo neighbors!')
                break
            # No infinite loops
            elif iteration > 500:
                self.logger.debug('\tMax iter')
                break
            elif np.isinf(num_neighbors):
                num_samples_to_go = min(10, self._continous_param_neighbor_samples - number_of_sampled_neighbors)
                if number_of_sampled_neighbors >= self._continous_param_neighbor_samples or num_samples_to_go <= 0:
                    break
                neighbors = hp.get_neighbors(array[index], self.rng,
                                             number=num_samples_to_go)
            else:
                if iteration > 0:
                    break
                neighbors = hp.get_neighbors(array[index], self.rng)
            # self.logger.debug('\t\t' + str(neighbors))
            # Check all newly obtained neighbors
            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    iteration += 1
                    continue
                new_array = array.copy()
                new_array = change_hp_value(self.incumbent.configuration_space, new_array, param, neighbor,
                                            index)
                try:
                    new_configuration = Configuration(self.incumbent.configuration_space, vector=new_array)
                    neighbourhood.append(new_configuration)
                    new_configuration.is_valid_configuration()
                    check_forbidden(self.cs.forbidden_clauses, new_array)
                    number_of_sampled_neighbors += 1
                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(new_configuration[param])
                except (ForbiddenValueError, ValueError) as e:
                    pass
                iteration += 1
        return checked_neighbors, checked_neighbors_non_unit_cube

    def _get_one_exchange_neighborhood_by_parameter(self):
        """
        Slight modification of ConfigSpace's get_one_exchange neighborhood. This orders the parameter values and samples
        more neighbors in one go. Further we need to rigorously check each and every neighbor if it is forbidden or not.
        """
        neighborhood_dict = {}
        params = list(self.incumbent.keys())
        self.logger.debug('params: ' + str(params))
        for index, param in enumerate(params):
            self.logger.info('Sampling neighborhood of %s' % param)
            array = self.incumbent.get_array()

            if not np.isfinite(array[index]):
                self.logger.info('>'.join(['-'*50, ' Not active!']))
                continue
            if self.old_sampling:
                checked_neighbors, checked_neighbors_non_unit_cube = self._old_sampling_of_one_exchange_neighborhood(
                    param, array, index
                )
            else:
                neighbourhood = []
                checked_neighbors = []
                checked_neighbors_non_unit_cube = []
                hp = self.incumbent.configuration_space.get_hyperparameter(param)
                num_neighbors = hp.get_num_neighbors(self.incumbent.get(param))
                self.logger.debug('\t' + str(num_neighbors))
                if num_neighbors == 0:
                    self.logger.debug('\tNo neighbors!')
                    continue
                elif np.isinf(num_neighbors):  # Continous Parameters
                    if hp.log:
                        base = np.e
                        log_lower = np.log(hp.lower) / np.log(base)
                        log_upper = np.log(hp.upper) / np.log(base)
                        neighbors = np.logspace(log_lower, log_upper, self._continous_param_neighbor_samples,
                                                endpoint=True, base=base)
                    else:
                        neighbors = np.linspace(hp.lower, hp.upper, self._continous_param_neighbor_samples)
                    neighbors = list(map(lambda x: hp._inverse_transform(x), neighbors))
                else:
                    neighbors = hp.get_neighbors(array[index], self.rng)
                for neighbor in neighbors:
                    if neighbor in checked_neighbors:
                        continue
                    new_array = array.copy()
                    new_array = change_hp_value(self.incumbent.configuration_space, new_array, param, neighbor,
                                                index)
                    try:
                        new_configuration = Configuration(self.incumbent.configuration_space, vector=new_array)
                        neighbourhood.append(new_configuration)
                        new_configuration.is_valid_configuration()
                        check_forbidden(self.cs.forbidden_clauses, new_array)
                        checked_neighbors.append(neighbor)
                        checked_neighbors_non_unit_cube.append(new_configuration[param])
                    except (ForbiddenValueError, ValueError) as e:
                        pass
            self.logger.info('>'.join(['-'*50, ' Found {:>3d} valid neighbors'.format(len(checked_neighbors))]))
            self._sampled_neighbors += len(checked_neighbors) + 1
            sort_idx = list(map(lambda x: x[0], sorted(enumerate(checked_neighbors), key=lambda y: y[1])))
            if isinstance(self.cs.get_hyperparameter(param), CategoricalHyperparameter):
                checked_neighbors_non_unit_cube = list(np.array(checked_neighbors_non_unit_cube)[sort_idx])
            else:
                checked_neighbors_non_unit_cube = np.array(checked_neighbors_non_unit_cube)[sort_idx]
            neighborhood_dict[param] = [np.array(checked_neighbors)[sort_idx], checked_neighbors_non_unit_cube]
        return neighborhood_dict

    def run(self) -> OrderedDict:
        """
        Main function.

        Returns
        -------
        evaluated_parameter_importance:OrderedDict
            Parameter -> importance. The order is important as smaller indices indicate higher importance
        """
        neighborhood_dict = self._get_one_exchange_neighborhood_by_parameter()  # sampled on a unit-hypercube!
        self.neighborhood_dict = neighborhood_dict
        incumbent_array = self.incumbent.get_array()
        def_perf, def_var = self._predict_over_instance_set(impute_inactive_values(self.cs.get_default_configuration()))
        inc_perf, inc_var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
        delta = def_perf - inc_perf
        evaluated_parameter_importance = {}

        # These are used for plotting and hold the predictions for each neighbor of each parameter
        # That means performance_dict holds the mean, variance_dict the variance of the forest
        performance_dict = {}
        variance_dict = {}
        # This are used for importance and hold the corresponding importance/variance over neighbors
        # Only import if NOT quantifying importance via performance-variance across neighbours
        overall_imp = {}
        # Nested list of values per tree in random forest
        pred_per_tree = {}

        # Iterate over parameters
        pbar = tqdm(range(self._sampled_neighbors), ascii=True, disable=not self.verbose)
        for index, param in enumerate(self.incumbent.keys()):
            if param not in neighborhood_dict:
                pbar.set_description('{: >.70s}'.format('Parameter %s is inactive' % param))
                continue

            pbar.set_description('Predicting performances for neighbors of {: >.30s}'.format(param))
            performance_dict[param] = []
            variance_dict[param] = []
            pred_per_tree[param] = []
            added_inc = False
            inc_at = 0
            # Iterate over neighbors
            for unit_neighbor, neighbor in zip(neighborhood_dict[param][0], neighborhood_dict[param][1]):
                if not added_inc:
                    # Detect incumbent
                    if unit_neighbor > incumbent_array[index]:
                        performance_dict[param].append(inc_perf)
                        variance_dict[param].append(inc_var)
                        pbar.update(1)
                        added_inc = True
                    else:
                        inc_at += 1
                # self.logger.debug('%s -> %s' % (self.incumbent[param], neighbor))
                # Create the neighbor-Configuration object
                new_array = incumbent_array.copy()
                new_array = change_hp_value(self.incumbent.configuration_space, new_array,
                                            param, unit_neighbor, index)
                new_configuration = impute_inactive_values(Configuration(self.incumbent.configuration_space,
                                                                         vector=new_array))
                # Predict performance
                x = np.array(new_configuration.get_array())
                pred_per_tree[param].append([np.mean(tree_pred) for tree_pred in self.model.rf.all_leaf_values(x)])
                # self.logger.debug("Pred per tree: %s", str(pred_per_tree[param][-1]))
                performance_dict[param].append(np.mean(pred_per_tree[param][-1]))
                variance_dict[param].append(np.var(pred_per_tree[param][-1]))

                pbar.update(1)
            if len(neighborhood_dict[param][0]) > 0:
                neighborhood_dict[param][0] = np.insert(neighborhood_dict[param][0], inc_at, incumbent_array[index])
                neighborhood_dict[param][1] = np.insert(neighborhood_dict[param][1], inc_at, self.incumbent[param])
            else:
                neighborhood_dict[param][0] = np.array(incumbent_array[index])
                neighborhood_dict[param][1] = [self.incumbent[param]]
            if not added_inc:
                mean, var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
                performance_dict[param].append(mean)
                variance_dict[param].append(var)
                pbar.update(1)
            # After all neighbors are estimated, look at all performances except the incumbent
            tmp_perf = performance_dict[param][:inc_at] + performance_dict[param][inc_at + 1:]
            if delta == 0:
                delta = 1  # To avoid division by zero
            imp_over_mea = (np.mean(tmp_perf) - performance_dict[param][inc_at]) / delta
            imp_over_med = (np.median(tmp_perf) - performance_dict[param][inc_at]) / delta
            try:
                imp_over_max = (np.max(tmp_perf) - performance_dict[param][inc_at]) / delta
            except ValueError:
                imp_over_max = np.nan  # Hacky fix as this is never used anyway
            overall_imp[param] = np.array([imp_over_mea, imp_over_med, imp_over_max])

        # Creating actual importance value (by normalizing over sum of vars)
        num_trees = len(list(pred_per_tree.values())[0][0])
        params = list(performance_dict.keys())
        overall_var_per_tree = {
                param : [np.var([neighbor[tree_idx] for neighbor in pred_per_tree[param]])
                            for tree_idx in range(num_trees)]
                                for param in params}
        # Sum up variances per tree across parameters
        sum_var_per_tree = [sum([overall_var_per_tree[param][tree_idx] for param in params])
                                     for tree_idx in range(num_trees)]
        # Normalize
        overall_var_per_tree = {p : [t / sum_var_per_tree[idx] for idx, t in enumerate(trees)] for p, trees in
                                                    overall_var_per_tree.items()}
        self.logger.debug("overall_var_per_tree %s (%d trees)",
                str(overall_var_per_tree), len(list(pred_per_tree.values())[0][0]))
        self.logger.debug("sum_var_per_tree %s (%d trees)",
                str(sum_var_per_tree), len(list(pred_per_tree.values())[0][0]))
        for param in performance_dict.keys():
            if self.quantify_importance_via_variance:
                evaluated_parameter_importance[param] = np.mean(overall_var_per_tree[param])
            else:
                evaluated_parameter_importance[param] = overall_imp[param][0]

        only_show = sorted(list(evaluated_parameter_importance.keys()),
                           key=lambda p: evaluated_parameter_importance[p])[:min(self.to_evaluate,
                                                                                 len(evaluated_parameter_importance.keys()))]

        self.neighborhood_dict = neighborhood_dict
        self.performance_dict = performance_dict
        self.variance_dict = variance_dict
        self.evaluated_parameter_importance = OrderedDict([(p, evaluated_parameter_importance[p]) for p in only_show])
        if self.quantify_importance_via_variance:
            self.evaluated_parameter_importance_uncertainty = OrderedDict([(p, np.std(overall_var_per_tree[p])) for p in only_show])
        all_res = {'imp': self.evaluated_parameter_importance,
                   'order': list(self.evaluated_parameter_importance.keys())}
        return all_res

    def _predict_over_instance_set(self, config):
        """
        Small wrapper to predict marginalized over instances
        Parameter
        ---------
        config:Configuration
            The self.incumbent of wich the performance across the whole instance set is to be estimated
        Returns
        -------
        mean
            the mean performance over the instance set
        var
            the variance over the instance set. If logged values are used, the variance might not be able to be used
        """
        mean, var = self.model.predict_marginalized_over_instances(np.array([config.get_array()]))
        return mean.squeeze(), var.squeeze()

    def plot_result(self, name='incneighbor', show=True):
        if not os.path.exists(name):
            os.mkdir(name)
        keys = deepcopy(list(self.incumbent.keys()))
        pbar = tqdm(list(keys), ascii=True, disable=not self.verbose)
        y_label = self.scenario.run_obj if self.scenario.run_obj != 'quality' else 'cost'
        for param in pbar:
            pbar.set_description('Plotting results for %s' % param)
            if param in self.performance_dict:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=.95)
                p, v = self.performance_dict[param], self.variance_dict[param]

                std = np.array(list(map(lambda s: np.sqrt(s), v))).flatten()
                upper = np.array(list(map(lambda x, y: x + y, p, std))).flatten()
                if self.scenario.run_obj == "runtime":
                    lower = np.array(list(map(lambda x, y: max(x - y, 0), p, std))).squeeze()
                else:
                    lower = np.array(list(map(lambda x, y: x - y, p, std))).squeeze()
                min_y = min(lower)
                max_y = max(upper)
                if not isinstance(self.incumbent.configuration_space.get_hyperparameter(param),
                                  CategoricalHyperparameter):
                    ax1.fill_between(self.neighborhood_dict[param][1], lower, upper, label='std', color=self.area_color)
                    ax1.plot(self.neighborhood_dict[param][1], p, label='predicted %s' % y_label,
                             ls='-', zorder=80,
                             **self.LINE_FONT)
                    label = True
                    c_inc = True
                    for n_idx, neighbor in enumerate(self.neighborhood_dict[param][1]):
                        if neighbor == self.incumbent[param] and c_inc:
                            ax1.scatter(neighbor, p[n_idx], label='incumbent', c='r', marker='.', zorder=999)
                            c_inc = False
                        elif self.show_query_points:
                            if label:
                                ax1.scatter(neighbor, p[n_idx], label='query points', c='w', marker='.',
                                            zorder=100, edgecolors='k')
                                label = False
                            else:
                                ax1.scatter(neighbor, p[n_idx], c='w', marker='.', zorder=100, edgecolors='k')
                    ax1.xaxis.grid(True)
                    ax1.yaxis.grid(True)
                    ax1.legend()
                else:
                    b = ax1.boxplot([[x] for x in p], showfliers=False)
                    plt.xticks(np.arange(1, len(self.neighborhood_dict[param][1])+1, 1),
                               self.neighborhood_dict[param][1])
                    # blow up boxes
                    for box, std_ in zip(b["boxes"], std):
                        y = box.get_ydata()
                        y[2:4] = y[2:4] + std_
                        if self.scenario.run_obj == "runtime":
                            y[0] = max(y[0] - std_, 0)
                            y[1] = max(y[1] - std_, 0)
                            y[4] = max(y[4] - std_, 0)
                        else:
                            y[0:2] = y[0:2] - std_
                            y[4] = y[4] - std_
                        box.set_ydata(y)
                        if self.scenario.run_obj == "runtime":
                            min_y = min(min_y, max(y[0] - std_, -0.01))
                        else:
                            min_y = min(min_y, y[0] - std_)
                        max_y = max(max_y, y[2] + std_)
                    for idx, t in enumerate(ax1.xaxis.get_ticklabels()):
                        color_ = (0., 0., 0.)
                        if self.neighborhood_dict[param][1][idx] == self.incumbent[param]:
                            color_ = (1, 0, 0)
                        t.set_color(color_)

                if isinstance(self.cs.get_hyperparameter(param), NumericalHyperparameter):
                    ax1.set_xscale('log' if self.cs.get_hyperparameter(param).log else 'linear')
                plt.xlabel(param)
                if self.scenario.run_obj == 'runtime':
                    ax1.set_ylabel('runtime [sec]', zorder=81, **self.LABEL_FONT)
                else:
                    ax1.set_ylabel('%s' % y_label, zorder=81, **self.LABEL_FONT)
                try:
                    plt.tight_layout()
                except ValueError:
                    pass
                ax1.set_ylim([min_y * 0.95, max_y])
                plt.savefig(os.path.join(name, param + '.png'))
                ax1.set_yscale('log')
                if min_y <= 0:
                    min_y = list(filter(lambda x: x > 0, lower))
                    if min_y:
                        min_y = min(min_y)
                    else:
                        min_y = 10**-3
                ax1.set_ylim([min_y * 0.95, max_y])
                plt.savefig(os.path.join(name, param + '_log.png'))
                plt.close('all')

    def plot_bokeh(self, plot_name=None, show_plot=False):
        """
        Plot local parameter importance in an interactive bokeh-plot.

        Parameters
        ----------
        plot_name: str
            path where to store the plot, None to not save it
        show_plot: bool
            whether or not to open plot in standard browser

        Returns
        -------
        layout: bokeh.models.Row
            bokeh plot (can be used in notebook or comparted with components)
        """
        plots = []
        pbar = tqdm(deepcopy(list(self.incumbent.keys())), ascii=True, disable=not self.verbose)
        for param in pbar:
            pbar.set_description('Plotting results (in bokeh) for %s' % param)
            if param in self.performance_dict:
                # Data to be plotted:
                p, v = self.performance_dict[param], self.variance_dict[param]
                std = np.array(list(map(lambda s: np.sqrt(s), v))).flatten()

                # Find incumbent indices
                inc_indices = [n_idx for n_idx, neighbor in enumerate(self.neighborhood_dict[param][1])
                               if neighbor == self.incumbent[param]]

                # Boxplot if categorical, else line
                if isinstance(self.incumbent.configuration_space.get_hyperparameter(param), CategoricalHyperparameter):
                    p = bokeh_boxplot(self.neighborhood_dict[param][1], p, std,
                                      y_label="runtime [sec]" if self.scenario.run_obj == "runtime" else "cost",
                                      x_label=param,
                                      runtime=self.scenario.run_obj=="runtime",
                                      inc_indices=inc_indices)
                else:
                    p = bokeh_line_uncertainty(self.neighborhood_dict[param][1], p, std,
                                               self.cs.get_hyperparameter(param).log,
                                               y_label="runtime [sec]" if self.scenario.run_obj == "runtime" else "cost",
                                               x_label=param,
                                               inc_indices=inc_indices)

            # Add as Panel so it can be easily put into Tabs
            plots.append(Panel(child=Row(p), title=param))

        # Putting the Tabs together
        layout = Tabs(tabs=[*plots])

        save_and_show(plot_name, show_plot, layout)

        return layout
