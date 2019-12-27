import copy
from collections import OrderedDict

import matplotlib as mpl
import numpy as np
from bokeh.models import ColumnDataSource, CDSView, IndexFilter, Span, BasicTickFormatter, Row
from bokeh.plotting import figure

from pimp.utils.bokeh_helpers import save_and_show, shorten_unique

mpl.use('Agg')
from matplotlib import pyplot as plt

from pimp.configspace import AndConjunction, Configuration, OrConjunction, impute_inactive_values
from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class Ablation(AbstractEvaluator):

    """
    Implementation of Ablation via surrogates
    """

    def __init__(self, scenario, cs, model, to_evaluate: int, rng, incumbent=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'Ablation'
        self.logger = 'pimp.' + self.name

        self.target = incumbent
        self.source = self.cs.get_default_configuration()
        self.target_active = {}
        self.source_active = {}
        self.delta, self.active = self._diff_in_source_and_target()
        self.default = self.cs.get_default_configuration()
        self._determine_combined_flipps()
        self.inactive = []

        params = self.cs.get_hyperparameters()
        self.n_params = len(params)
        self.n_feats = len(self.types) - len(params)
        self.insts = copy.deepcopy(self.scenario.train_insts)
        if len(self.scenario.test_insts) > 1:
            self.insts.extend(self.scenario.test_insts)
        self.predicted_parameter_performances = OrderedDict()
        self.predicted_parameter_variances = OrderedDict()

########################################################################################################################
    # HANDLING FORBIDDENS # HANDLING FORBIDDENS # HANDLING FORBIDDENS # HANDLING FORBIDDENS # HANDLING FORBIDDENS
########################################################################################################################
    def determine_forbidden(self):
        """
        Method to determine forbidden clauses, and saves them in a simple to check format.
        return: list of lists [[pname, pvalue, pname2, pvalue2]]
        """
        forbidden_clauses = self.cs.forbidden_clauses
        forbidden_descendants = map(lambda x: x.get_descendant_literal_clauses(), forbidden_clauses)
        forbidden_names_value_paris = list()
        for forbidden_literals in forbidden_descendants:
            elem = []
            for literal in forbidden_literals:
                elem.append(literal.hyperparameter.name)
                elem.append(literal.value)
            forbidden_names_value_paris.append(elem)
        return forbidden_names_value_paris

    def check_not_forbidden(self, forbidden_name_value_pairs, modifiable_config):
        """
        Helper method to determine if a current configuration dictionary is forbidden or not
        forbidden_name_value_pairs: list of lists of forbidden parameter settings
        modifiable_config: dict param_name -> param_value
        return: boolean. True if the current configuration is not forbidden, False otherwise.
        """
        not_forbidden = True
        for forbidden_clause in forbidden_name_value_pairs:
            sum_forbidden = 0
            for key in modifiable_config:
                if key in forbidden_clause:
                    at_ = forbidden_clause.index(key) + 1
                    if modifiable_config[key] == forbidden_clause[at_]:
                        sum_forbidden += 2
            not_forbidden = not_forbidden and not (sum_forbidden == len(forbidden_clause))
            if not not_forbidden:
                return False
        return True

########################################################################################################################
    # HANDLING CONDITIONALITIES # HANDLING CONDITIONALITIES # HANDLING CONDITIONALITIES # HANDLING CONDITIONALITIES
########################################################################################################################
    def _helper(self, modified_delta):
        to_remove = []
        indices_to_check = [i for i in range(len(modified_delta))]
        for at, idx in enumerate(indices_to_check):
            parameters = modified_delta[idx]
            for p in parameters:
                children = self.cs.get_children_of(p)
                for child in children:
                    if child.name not in modified_delta[idx]:
                        for condition in self.cs.get_parent_conditions_of(child):
                            if type(condition) in [AndConjunction, OrConjunction]:  # Case where parents of parents are
                                                                                    # not set get checked here
                                skip_this = False
                                for component in condition.components:
                                    if self.source.get(component.parent.name) is None:
                                        if [child.name] in modified_delta:
                                            if modified_delta.index([child.name]) not in to_remove:
                                                to_remove.append(modified_delta.index([child.name]))
                                        for tmp_idx, d in enumerate(modified_delta):
                                            if component.parent.name in d and child.name not in d:
                                                modified_delta[tmp_idx].append(child.name)
                                                indices_to_check.append(tmp_idx)
                                        skip_this = True
                                    if self.target.get(component.parent.name) is None:
                                        if [child.name] in modified_delta:
                                            if modified_delta.index([child.name]) not in to_remove:
                                                to_remove.append(modified_delta.index([child.name]))
                                        for tmp_idx, d in enumerate(modified_delta):
                                            if component.parent.name in d and child.name not in d:
                                                modified_delta[tmp_idx].append(child.name)
                                                indices_to_check.append(tmp_idx)
                                        skip_this = True
                                if skip_this:
                                    continue
                            try:
                                source_conditions_met = condition.evaluate(self.source)
                            except ValueError:
                                source_conditions_met = False
                            try:
                                target_conditions_met = condition.evaluate(self.target)
                            except ValueError:
                                target_conditions_met = False
                            if target_conditions_met and not source_conditions_met:
                                if child.name not in modified_delta[idx]:
                                    modified_delta[idx].append(child.name)  # Now at idx delta has two combined entries
                                    indices_to_check.append(idx)
                                try:
                                    tmp_idx = modified_delta.index([child.name])
                                    if [child.name] in modified_delta and tmp_idx not in to_remove:
                                        to_remove.append(modified_delta.index([child.name]))
                                except ValueError:
                                    pass
        return to_remove, modified_delta

    def _determine_combined_flipps(self):
        """
        Method to determine parameters that have to be jointly flipped with their parents.
        Uses the methods provided by Config space to easily check conditions
        """
        to_remove, self.delta = self._helper(self.delta)
        to_remove = sorted(to_remove, reverse=True)  # reverse sort necessary to not delete the wrong items
        for idx in to_remove:
            self.delta.pop(idx)
        to_remove = []
        single_remove = []
        for i in range(len(self.delta) - 1):
            flip_1 = self.delta[i]
            for j in range(i + 1, len(self.delta)):
                flip_2 = self.delta[j]
                for idx, entry in enumerate(flip_2):
                    if entry in flip_1:
                        if idx == 0:
                            to_remove.append(j)
                            self.logger.info('Removing %s' % str(self.delta[j]))
                        else:
                            single_remove.append((j, idx))
        single_remove = sorted(single_remove, reverse=True)
        for tuple_ in single_remove:
            del self.delta[tuple_[0]][tuple_[1]]
        to_remove = sorted(to_remove, reverse=True)  # reverse sort necessary to not delete the wrong items
        for idx in to_remove:
            self.delta.pop(idx)

    def _check_child_conditions(self, _dict, children):
        dict_ = {}
        for child in children:
            all_child_conditions_fulfilled = True
            for condition in self.cs.get_parent_conditions_of(child):
                try:
                    all_child_conditions_fulfilled = all_child_conditions_fulfilled and condition.evaluate(_dict)
                except ValueError:
                    all_child_conditions_fulfilled = False  # Parent not set!
            dict_[child.name] = all_child_conditions_fulfilled
        return dict_

    def _check_children(self, modded_dict, params, delete=False):
        """
        Method that checks if children are set to inactive during the current round due to condidionalities.

        Parameters
        ----------
        modded_dict: dictionary
            Dictionary with parameter_name -> parameter_value
        params: list
            list of parameters to check the child activity for
        delete: boolean
            allows for deletion of inactive parameters from the ablation_delta_list

        Returns
        -------
        modded_dict, where inactive parameters have the value None
        """
        for param in params:
            children = self.cs.get_children_of(param)
            if children:
                child_conditions = self._check_child_conditions(modded_dict, children)
                for child in child_conditions:
                    if not child_conditions[child]:
                        modded_dict[child] = None
                        self.logger.debug('Deactivated child %s found this round' % child)
                        modded_dict = self._set_child_of_child_to_none(child, modded_dict)
                        if delete and [child] in self.delta:
                            for item in self.delta:
                                self.logger.debug('%s, %s' % (str([child]), str(item)))
                            self.logger.critical('Removing deactivated parameter %s' % child)
                            self.delta.pop(self.delta.index([child]))
                    else:
                        if child not in self.target_active and child not in self.source_active:
                            modded_dict[child] = None
                        elif child in self.target_active:
                            if not self.target_active[child]:
                                if child in self.source_active:
                                    modded_dict[child] = self.source[child]
                                else:
                                    modded_dict[child] = None
                            elif self.target_active[child]:
                                modded_dict[child] = self.target[child]
                            else:
                                modded_dict[child] = self.cs.get_hyperparameter(child).default_value
                        modded_dict = self._check_children(modded_dict, [child])
        return modded_dict

    def _set_child_of_child_to_none(self, child, modded_dict):
        children_of_child = self.cs.get_children_of(child)
        if children_of_child:
            modded_dict[child] = self.target[child]
            try:
                conditions = self._check_child_conditions(modded_dict, children_of_child)
            except ValueError:
                modded_dict[child] = None
                return modded_dict
            modded_dict[child] = None
            for c in conditions:
                if not conditions[c] and c in list(modded_dict.keys()):
                    modded_dict[c] = None
                    self.logger.debug('Deactivated sub-child %s found this round' % c)
                    modded_dict = self._set_child_of_child_to_none(c, modded_dict)
        return modded_dict

    def _rm_inactive(self, candidates, modifiable_config_dict, prev_modifiable_config_dict, removed=[]):
        for candidate in candidates:
            parent_conditions = self.cs.get_parent_conditions_of(candidate)
            passing = True
            for condition in parent_conditions:
                try:
                    passing = condition.evaluate(modifiable_config_dict) and passing
                except ValueError:
                    passing = False
            if not passing:
                if candidate in modifiable_config_dict:
                    try:
                        modifiable_config_dict[candidate] = prev_modifiable_config_dict[candidate]
                    except KeyError:
                        del modifiable_config_dict[candidate]
                    removed.append(candidate)
                modifiable_config_dict, removed = self._rm_inactive(
                    list(map(lambda x: x.name, self.cs.get_children_of(candidate))), modifiable_config_dict,
                    prev_modifiable_config_dict, removed)
        return modifiable_config_dict, removed

########################################################################################################################
    # MAIN METHOD # MAIN METHOD # MAIN METHOD # MAIN METHOD # MAIN METHOD # MAIN METHOD # MAIN METHOD # MAIN METHOD
########################################################################################################################
    def run(self) -> OrderedDict:
        """
        Main function.
        Returns
        -------
        evaluated_parameter_importance:OrderedDict
            Parameter -> importance. The order is important as smaller indices indicate higher importance
        """
        # Minor setup
        modifiable_config_dict = copy.deepcopy(self.source.get_dictionary())
        prev_modifiable_config_dict = copy.deepcopy(self.source.get_dictionary())
        modified_so_far = []
        start_delta = len(self.delta)
        best_performance = -1

        # Predict source and target performance to later use it to predict the %improvement a parameter causes
        source_mean, source_var = self._predict_over_instance_set(impute_inactive_values(self.source))
        prev_performance = source_mean
        target_mean, target_var = self._predict_over_instance_set(impute_inactive_values(self.target))
        improvement = prev_performance - target_mean
        self.predicted_parameter_performances['-source-'] = source_mean.flatten()[0]
        self.predicted_parameter_variances['-source-'] = source_var.flatten()[0]
        self.evaluated_parameter_importance['-source-'] = 0

        forbidden_name_value_pairs = self.determine_forbidden()
        length_ = len(self.delta) - min(len(self.delta), self.to_evaluate)
        self.logger.info('Difference in source and target: %d' % len(self.delta))

        while len(self.delta) > length_:  # Main loop. While parameters still left ...
            modifiable_config_dict = copy.deepcopy(prev_modifiable_config_dict)
            self.logger.debug('Round %d of %d:' % (start_delta - len(self.delta) + 1, min(start_delta,
                                                                                          self.to_evaluate)))
            for param_tuple in modified_so_far:  # necessary due to combined flips
                for parameter in param_tuple:
                    modifiable_config_dict[parameter] = self.target[parameter]
            prev_modifiable_config_dict = copy.deepcopy(modifiable_config_dict)

            round_performances = []
            round_variances = []
            for candidate_tuple in self.delta:
                self.logger.debug('candidate(s): %s' % str(candidate_tuple))

                for candidate in candidate_tuple:
                    self.logger.debug(' {:<25s} -> {:^10s}'.format(str(candidate), str(self.target[candidate])))
                    modifiable_config_dict[candidate] = self.target[candidate]

                modifiable_config_dict = self._check_children(modifiable_config_dict, candidate_tuple)

                # Check if current config is allowed
                not_forbidden = self.check_not_forbidden(forbidden_name_value_pairs, modifiable_config_dict)
                if not not_forbidden:  # othwerise skipp it
                    self.logger.critical('FOUND FORBIDDEN!!!!! SKIPPING!!!')
                    continue
                try:
                    modifiable_config = Configuration(self.cs, modifiable_config_dict)
                except ValueError:
                    modifiable_config_dict, rmed = self._rm_inactive(candidate_tuple[1:], modifiable_config, prev_modifiable_config_dict)
                    modifiable_config = Configuration(self.cs, modifiable_config_dict)

                mean, var = self._predict_over_instance_set(impute_inactive_values(modifiable_config))  # ... predict their performance
                self.logger.debug('%s: %.6f' % (candidate_tuple, mean[0]))
                round_performances.append(mean)
                round_variances.append(var)
                modifiable_config_dict = copy.deepcopy(prev_modifiable_config_dict)

            best_idx = np.argmin(round_performances)
            assert 0 <= best_idx < len(round_performances), 'No improving parameter found!'
            best_performance = round_performances[best_idx]  # greedy choice of parameter to fix
            best_variance = round_variances[best_idx]
            improvement_in_percentage = (prev_performance - best_performance) / improvement
            prev_performance = best_performance
            modified_so_far.append(self.delta[best_idx])
            self.logger.info('Round %2d winner(s): (%s, %.4f)' % (start_delta - len(self.delta) + 1,
                                                                  str(self.delta[best_idx]),
                                                                  improvement_in_percentage * 100))
            param_str = '; '.join(self.delta[best_idx])
            self.evaluated_parameter_importance[param_str] = improvement_in_percentage.flatten()[0]
            self.predicted_parameter_performances[param_str] = best_performance.flatten()[0]
            self.predicted_parameter_variances[param_str] = best_variance.flatten()[0]

            for winning_param in self.delta[best_idx]:  # Delete parameters that were set to inactive by the last
                                                        # best parameter

                prev_modifiable_config_dict[winning_param] = self.target[winning_param]
            self._check_children(prev_modifiable_config_dict, self.delta[best_idx], delete=True)
            self.delta.pop(best_idx)  # don't forget to remove already tested parameters

        self.predicted_parameter_performances['-target-'] = target_mean.flatten()[0]
        self.predicted_parameter_variances['-target-'] = target_var.flatten()[0]
        self.evaluated_parameter_importance['-target-'] = 0
        # sum_ = 0  # Small check that sum is 1
        # for key in self.evaluated_parameter_importance:
        #     print(key, self.evaluated_parameter_importance[key])
        #     sum_ += self.evaluated_parameter_importance[key]
        # print(sum_)
        all_res = {'perf': self.predicted_parameter_performances, 'var': self.predicted_parameter_variances,
                'imp': self.evaluated_parameter_importance, 'order': list(self.evaluated_parameter_importance.keys())}
        return all_res

########################################################################################################################
# HELPER METHODS # HELPER METHODS # HELPER METHODS # HELPER METHODS # HELPER METHODS # HELPER METHODS # HELPER METHODS
########################################################################################################################
    def _predict_over_instance_set(self, config):
        """
        Small wrapper to predict marginalized over instances
        Parameter
        ---------
        config:Configuration
            The configuration of wich the performance across the whole instance set is to be estimated
        Returns
        -------
        mean
            the mean performance over the instance set
        var
            the variance over the instance set. If logged values are used, the variance might not be able to be used
        """
        mean, var = self.model.predict_marginalized_over_instances(np.array([config.get_array()]))
        return mean, var

    def _diff_in_source_and_target(self):
        """
        Helper Method to determine which parameters might lie on an ablation path
        Return
        ------
        delta:list
            List of parameters that are modified from the source to the target
        """
        delta = []
        active = {}
        for parameter in self.cs.get_hyperparameters():
            parameter = parameter.name
            tmp = ' not'
            if self.source[parameter] != self.target[parameter] and self.target[parameter] is not None:
                tmp = ''
                delta.append([parameter])
            self.logger.debug('%s was%s modified from source to target (%s, %s) [s, t]' % (parameter, tmp,
                                                                                           self.source[parameter],
                                                                                           self.target[parameter]))
            self.target_active[parameter] = True if self.target[parameter] is not None else False
            active[parameter] = True if self.source[parameter] is not None else False
        self.source_active = copy.deepcopy(active)
        return delta, active

########################################################################################################################
# PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING
########################################################################################################################
    def plot_result(self, name=None, show=True):
        self.plot_predicted_percentage(plot_name=name+'percentage.png', show=show)
        self.plot_predicted_performance(plot_name=name+'performance.png', show=show)
        self.logger.info('Saved plots as %s[percentage|performance].png' % name)
        plt.close('all')


    def plot_bokeh(self, plot_name=None, show_plot=False, predicted_percentage=True, predicted_performance=True):
        """
        Plot ablation results in an interactive bokeh-plot.

        Parameters
        ----------
        plot_name: str
            path where to store the plot, None to not save it
        show_plot: bool
            whether or not to open plot in standard browser
        predicted_percentage: bool
            include a barchart of individual parameter contributions of the improvement from source to target
        predicted_performance: bool
            include a plot of the ablation path using the predicted performances of parameter flips

        Returns
        -------
        layout: bokeh.models.Row
            bokeh plot (can be used in notebook or comparted with components)
        """

        # Get all relevant data-points
        p_names = list(self.evaluated_parameter_importance.keys())
        p_names_short = shorten_unique(p_names)
        p_importance = [100 * self.evaluated_parameter_importance[p] for p in p_names]
        p_performance = [self.predicted_parameter_performances[p] for p in p_names]
        p_variance = [self.predicted_parameter_variances[p] for p in p_names]

        # We don't always want to plot everything, reduce plotted parameters by taking out the least important ones
        max_to_plot = min(len(p_names), self.MAX_PARAMS_TO_PLOT)
        plot_indices = sorted(range(1, len(p_importance) - 1), key=lambda x: p_importance[x], reverse=True)[:max_to_plot]
        plot_indices = [0] + plot_indices + [len(p_importance) - 1]  # Always keep "source" and "target"

        # Customizing plot-style
        bar_width = 25
        tooltips = [("Parameter", "@parameter_names"),
                    ("Importance", "@parameter_importance"),
                    ("Predicted Performance", "@parameter_performance")]

        # Create ColumnDataSource for both plots
        source = ColumnDataSource(data=dict(parameter_names_short=p_names_short,
                                            parameter_names=p_names,
                                            parameter_importance=p_importance,
                                            parameter_performance=p_performance,
                                            parameter_variance=p_variance,
                                            ))

        plots = []
        if predicted_percentage:
            # Don't plot source and target in this plot, use IndexFilter
            view_predicted_percentage = CDSView(source=source, filters=[IndexFilter(plot_indices[1:-1])])
            p = figure(x_range=[p_names_short[idx] for idx in plot_indices[1:-1]],
                       plot_height=350, plot_width=100+max_to_plot*bar_width,
                       toolbar_location=None, tools="hover", tooltips=tooltips)

            p.vbar(x='parameter_names_short', top='parameter_importance', width=0.9,
                   source=source, view=view_predicted_percentage)
            for value in [self.IMPORTANCE_THRESHOLD, -self.IMPORTANCE_THRESHOLD]:
                p.add_layout(Span(location=value*100, dimension='width', line_color='red', line_dash='dashed'))

            p.yaxis.axis_label = "improvement [%]"
            plots.append(p)

        if predicted_performance:
            view_predicted_performance = CDSView(source=source, filters=[IndexFilter(plot_indices)])
            p = figure(x_range=[p_names_short[idx] for idx in plot_indices],
                       plot_height=350, plot_width=100 + max_to_plot * bar_width,
                       toolbar_location=None, tools="hover", tooltips=tooltips)

            p.line(x='parameter_names_short', y='parameter_performance', source=source, view=view_predicted_performance)
            band_x = np.append(list(range(len(plot_indices))), list(range(len(plot_indices)))[::-1])
            band_y = np.append([max(p-np.sqrt(v), 0) if self.scenario.run_obj == "runtime" else p-np.sqrt(v)
                                for p, v in [(p_performance[idx], p_variance[idx]) for idx in plot_indices]],
                               [p+np.sqrt(v) for p, v in [(p_performance[idx], p_variance[idx])
                                                          for idx in plot_indices][::-1]])
            p.patch(band_x, band_y, color='#7570B3', fill_alpha=0.2)

            p.yaxis.axis_label = "est. runtime [sec]" if self.scenario.run_obj == 'runtime' else 'est. cost'
            plots.append(p)

        # Common styling:
        for p in plots:
            p.xaxis.major_label_orientation = 1.3
            p.xaxis.major_label_text_font_size = "14pt"
            p.yaxis.formatter = BasicTickFormatter(use_scientific=False)

        layout = Row(*plots)

        # Save and show...
        save_and_show(plot_name, show_plot, layout)

        return layout

    def plot_predicted_percentage(self, plot_name=None, show=True):
        """
        Method to plot a barchart of individual parameter contributions of the improvement from source to target
        """
        p_names = np.array(list(self.evaluated_parameter_importance.keys())[1:-1])
        p_names_short = np.array(shorten_unique(p_names))
        p_importance = np.array([100 * self.evaluated_parameter_importance[p] for p in p_names])
        max_to_plot = min(len(p_names), self.MAX_PARAMS_TO_PLOT)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=.95)

        ax1.bar(list(range(len(p_names))), p_importance, color=self.area_color)

        ax1.set_xticks(np.arange(len(p_names)))
        ax1.set_xlim(-0.5, max_to_plot - .25)
        ax1.set_ylim(min(p_importance) - max(2, min(p_importance)*.1),
                     max(p_importance) + min(2, max(p_importance)*.1))

        ax1.set_ylabel('improvement [%]', zorder=81, **self.LABEL_FONT)
        ax1.plot(list(range(-1, len(p_names_short) + 1)),
                 [self.IMPORTANCE_THRESHOLD*100 for _ in range(len(p_names_short) + 2)], c='r', linestyle='--')
        ax1.plot(list(range(-1, len(p_names_short) + 1)),
                 [-self.IMPORTANCE_THRESHOLD*100 for _ in range(len(p_names_short) + 2)], c='r', linestyle='--')
        ax1.set_xticklabels(p_names_short, rotation=25, ha='right', **self.AXIS_FONT)

        for idx, t in enumerate(ax1.xaxis.get_ticklabels()):
            color_ = (0.45, 0.45, 0.45)
            if self.evaluated_parameter_importance[p_names[idx]] > self.IMPORTANCE_THRESHOLD:
                color_ = (0., 0., 0.)
            t.set_color(color_)

        ax1.xaxis.grid(True)
        ax1.yaxis.grid(True)
        try:
            plt.tight_layout()
        except ValueError:
            pass

        if plot_name is not None:
            fig.savefig(plot_name)
            if show:
                plt.show()
        else:
            plt.show()

    def plot_predicted_performance(self, plot_name=None, show=True):
        """
        Method to plot the ablation path using the predicted performances of parameter flips
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=.95)
        y_label = self.scenario.run_obj if self.scenario.run_obj != 'quality' else 'cost'

        path = list(self.predicted_parameter_performances.keys())
        path = np.array(shorten_unique(path))
        max_to_plot = min(len(path), self.MAX_PARAMS_TO_PLOT)
        performances = list(self.predicted_parameter_performances.values())
        performances = np.array(performances).reshape((-1, 1))
        variances = list(self.predicted_parameter_variances.values())
        variances = np.array(variances).reshape((-1, 1))

        if max_to_plot == self.MAX_PARAMS_TO_PLOT:
            performances[self.MAX_PARAMS_TO_PLOT - 1] = performances[-1]
            variances[self.MAX_PARAMS_TO_PLOT - 1] = variances[-1]
            path[self.MAX_PARAMS_TO_PLOT - 1] = path[-1]

        ax1.plot(list(range(len(performances))), performances,
                 label='predicted %s' % y_label,
                 ls='-', zorder=80,
                 **self.LINE_FONT)

        upper = np.array(list(map(lambda x, y: x + np.sqrt(y), performances, variances))).flatten()
        if self.scenario.run_obj == "runtime":
            lower = np.array(list(map(lambda x, y: max(x - np.sqrt(y), np.array([0])), performances,
                                      variances))).squeeze()
        else:
            lower = np.array(list(map(lambda x, y: x - np.sqrt(y), performances,
                                      variances))).squeeze()
        ax1.fill_between(list(range(len(performances))), lower, upper, label='std', color=self.area_color)

        ax1.set_xticks(list(range(len(path))))
        ax1.set_xticklabels(path, rotation=25, ha='right', **self.AXIS_FONT)
        percentages = list(self.evaluated_parameter_importance.values())
        for idx, t in enumerate(ax1.xaxis.get_ticklabels()):
            color_ = (0.45, 0.45, 0.45)
            if percentages[idx] > self.IMPORTANCE_THRESHOLD:
                color_ = (0., 0., 0.)
            t.set_color(color_)

        ax1.set_xlim(0, max_to_plot - 1)
        if self.scenario.run_obj == "runtime":
            ax1.set_ylim(max(min(lower) - max(.1 * min(lower), 0.1), 0), max(upper) + .1 * max(upper))
        else:
            ax1.set_ylim(min(lower) - max(.1 * min(lower), 0.1), max(upper) + .1 * max(upper))

        ax1.legend()
        if self.scenario.run_obj == 'runtime':
            ax1.set_ylabel('runtime [sec]', zorder=81, **self.LABEL_FONT)
        else:
            ax1.set_ylabel('%s' % y_label, zorder=81, **self.LABEL_FONT)
        ax1.xaxis.grid(True)
        ax1.yaxis.grid(True)
        try:
            plt.tight_layout()
        except ValueError:
            pass

        if plot_name is not None:
            fig.savefig(plot_name)
            if show:
                plt.show()
        else:
            plt.show()
