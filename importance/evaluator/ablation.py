from collections import OrderedDict
from importance.configspace import Configuration
import numpy as np
from importance.evaluator.base_evaluator import AbstractEvaluator
import copy
from matplotlib import pyplot as plt


class Ablation(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, incumbent=None, logy=True,
                 target_performance=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'Ablation'
        self.logger = self.name

        self.target = incumbent
        self.source = self.cs.get_default_configuration()
        self.delta = self._diff_in_source_and_target()
        self._determine_combined_flipps()

        params = self.cs.get_hyperparameters()
        self.n_params = len(params)
        self.n_feats = len(self.types) - len(params)
        self.insts = copy.deepcopy(self.scenario.train_insts)
        self.logy = logy
        if len(self.scenario.test_insts) > 1:
            self.insts.extend(self.scenario.test_insts)
        self.target_performance = target_performance
        self.predicted_parameter_performances = OrderedDict()

    def _diff_in_source_and_target(self):
        delta = []
        for parameter in self.source:
            tmp = ' not'
            if self.source[parameter] != self.target[parameter]:
                tmp = ''
                delta.append([parameter])
            self.logger.debug('%s was%s modified from source to target (%s, %s) [s, t]' % (parameter, tmp,
                                                                                           self.source[parameter],
                                                                                           self.target[parameter]))
        return delta

    def _determine_combined_flipps(self):
        to_remove = []
        for idx, parameter in enumerate(self.delta):
            children = self.cs.get_children_of(parameter[0])
            for child in children:
                for condition in self.cs.get_parent_conditions_of(child):
                    if condition.evaluate(self.target) and not condition.evaluate(self.source):
                        self.delta[idx].append(child)  # Now at idx delta has two combined entries
                        if [child] in self.delta:
                            to_remove.append(self.delta.index([child]))
        to_remove = sorted(to_remove, reverse=True)  # reverse sort necessary to not delete the wrong items
        for idx in to_remove:
            self.delta.pop(idx)

    def run(self) -> OrderedDict:
        modifiable_config_dict = copy.deepcopy(self.source.get_dictionary())
        modified_so_far = []
        start_delta = len(self.delta)
        best_performance = -1

        source_mean, var = self._predict_over_instance_set(self.source)
        prev_performance = source_mean
        target_mean, var = self._predict_over_instance_set(self.target)
        improvement = prev_performance - target_mean
        self.predicted_parameter_performances['-source-'] = source_mean
        self.evaluated_parameter_importance['-source-'] = 0

        while len(self.delta) > 0:
            self.logger.debug('Round %d of %d:' % (start_delta - len(self.delta), start_delta - 1))
            for param_tuple in modified_so_far:  # necessary due to combined flips
                for parameter in param_tuple:
                    modifiable_config_dict[parameter] = self.target[parameter]

            round_performances = []
            for candidate_tuple in self.delta:
                for candidate in candidate_tuple:
                    modifiable_config_dict[candidate] = self.target[candidate]
                modifiable_config = Configuration(self.cs, modifiable_config_dict)
                mean, var = self._predict_over_instance_set(modifiable_config)
                self.logger.debug('%s: %.6f' % (candidate_tuple, mean[0]))
                round_performances.append(mean)
                for candidate in candidate_tuple:
                    modifiable_config_dict[candidate] = self.source[candidate]

            best_idx = np.argmin(round_performances)
            best_performance = round_performances[best_idx]
            improvement_in_percentage = (prev_performance - best_performance) / improvement
            prev_performance = best_performance
            modified_so_far.append(self.delta[best_idx])
            self.logger.info('Round %2d winner(s): (%s, %.4f)' % (start_delta - len(self.delta),
                                                                  str(self.delta[best_idx]),
                                                                  improvement_in_percentage * 100))
            param_str = '; '.join(self.delta[best_idx])
            self.evaluated_parameter_importance[param_str] = improvement_in_percentage
            self.predicted_parameter_performances[param_str] = best_performance
            self.delta.pop(best_idx)
        self.predicted_parameter_performances['-target-'] = best_performance
        self.evaluated_parameter_importance['-target-'] = 0
        # sum_ = 0  # Small check that sum is 1
        # for key in self.evaluated_parameter_importance:
        #     print(key, self.evaluated_parameter_importance[key])
        #     sum_ += self.evaluated_parameter_importance[key]
        # print(sum_)
        return self.evaluated_parameter_importance

    def _predict_over_instance_set(self, config):
        mean, var = self.model.predict_marginalized_over_instances(np.array([config.get_array()]))
        if self.logy:
            mean = np.power(10, mean)
        return mean, var

    def plot_result(self, name=None, title='Surrogate-Ablation', fontsize=38, lw=6):
        self.plot_predicted_percentage(plot_name=name, title=title, fontsize=fontsize)
        self.plot_predicted_performance(plot_name=name, title=title, fontsize=fontsize, lw=lw)

    def plot_predicted_percentage(self, title='Surrogate-Ablation', plot_name=None, fontsize=38):
        fig = plt.figure()  # figsize=(14, 18))
        plt.subplots_adjust(bottom=0.25, top=0.7, left=0.05, right=.95)
        fig.suptitle(title, fontsize=fontsize)
        ax1 = fig.add_subplot(111)

        ax1.bar(list(range(len(self.evaluated_parameter_importance.keys()))),
                list(self.evaluated_parameter_importance.values()))

        path = list(self.predicted_parameter_performances.keys())
        path = np.array(path)

        ax1.set_xticks(list(range(len(path))))
        ax1.set_xlim(0, len(path) - 1)

        ax1.set_xticklabels(path, rotation=25, ha='right')

        plt.tight_layout()

        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            plt.show()

    def plot_predicted_performance(self, title='Surrogate-Ablation', plot_name=None, lw=6,
                    fontsize=38):
        color = (0.45, 0.45, 0.45)

        fig = plt.figure()  # figsize=(14, 18))
        fig.suptitle(title, fontsize=fontsize)
        ax1 = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.25, top=0.7, left=0.05, right=.95)

        path = list(self.predicted_parameter_performances.keys())
        path = np.array(path)
        performances = list(self.predicted_parameter_performances.values())
        performances = np.array(performances).reshape((24,))

        ax1.plot(list(range(len(performances))), performances, label='Predicted Performance',
                 color=color, ls='-', lw=lw, zorder=80)

        # ax1.set_xlabel(x_axis_lower_label + ' path', color=color, fontsize=fontsize)

        ax1.set_xticks(list(range(len(path))))
        ax1.set_xticklabels(path, rotation=25, ha='right', color=color)

        ax1.set_xlim(0, len(path) - 1)

        ax1.legend()
        ax1.set_ylabel('runtime [sec]', fontsize=fontsize, zorder=81)
        ax1.xaxis.grid(True)
        gl = ax1.get_xgridlines()
        for l in gl:
            l.set_linewidth(5)
        handles, labels = ax1.get_legend_handles_labels()

        # reverse the order
        ax1.legend(handles[::-1], labels[::-1])
        plt.tight_layout()

        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            plt.show()
