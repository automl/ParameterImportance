import os
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from pimp.configspace import change_hp_value, Configuration, ForbiddenValueError,\
    impute_inactive_values, CategoricalHyperparameter
from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class IncNeighbor(AbstractEvaluator):

    """
    Implementation of Ablation via surrogates
    """

    def __init__(self, scenario, cs, model, to_evaluate: int, incumbent=None, **kwargs):
        super().__init__(scenario, cs, model, to_evaluate, **kwargs)
        self.name = 'IncNeighbor'
        self.logger = self.name
        self.incumbent = incumbent
        self.incumbent._populate_values()  # type: Configuration
        self._continous_param_neighbor_samples = 100
        self.neighborhood_dict = None
        self.performance_dict = {}
        self.variance_dict = {}

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
            neighbourhood = []
            number_of_sampled_neighbors = 0
            array = self.incumbent.get_array()

            if not np.isfinite(array[index]):
                self.logger.info('>'.join(['-'*50, ' Not active!']))
                continue

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
                self.logger.debug('\t\t' + str(neighbors))
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
                        self.incumbent.configuration_space._check_forbidden(new_array)
                        number_of_sampled_neighbors += 1
                        checked_neighbors.append(neighbor)
                        checked_neighbors_non_unit_cube.append(new_configuration[param])
                    except ForbiddenValueError as e:
                        pass
                    iteration += 1
            self.logger.info('>'.join(['-'*50, ' Found {:>3d} valid neighbors'.format(len(checked_neighbors))]))
            sort_idx = list(map(lambda x: x[0], sorted(enumerate(checked_neighbors), key=lambda y: y[1])))
            neighborhood_dict[param] = [np.array(checked_neighbors)[sort_idx],
                                        np.array(checked_neighbors_non_unit_cube)[sort_idx]]
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
        # TODO get some kind of importance measure in this
        performance_dict = {}
        variance_dict = {}
        incumbent_array = self.incumbent.get_array()
        for index, param in enumerate(self.incumbent.keys()):
            if param in neighborhood_dict:
                self.logger.info('Predicting performances for neighbors of %s' % param)
                performance_dict[param] = []
                variance_dict[param] = []
                added_inc = False
                inc_at = 0
                for unit_neighbor, neighbor in zip(neighborhood_dict[param][0], neighborhood_dict[param][1]):
                    if not added_inc:
                        if unit_neighbor > incumbent_array[index]:
                            mean, var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
                            performance_dict[param].append(mean)
                            variance_dict[param].append(var)
                            added_inc = True
                        else:
                            inc_at += 1
                    self.logger.debug('%s -> %s' % (self.incumbent[param], neighbor))
                    new_array = incumbent_array.copy()
                    new_array = change_hp_value(self.incumbent.configuration_space, new_array, param, unit_neighbor,
                                                index)
                    new_configuration = impute_inactive_values(Configuration(self.incumbent.configuration_space,
                                                                             vector=new_array))
                    mean, var = self._predict_over_instance_set(new_configuration)
                    performance_dict[param].append(mean)
                    variance_dict[param].append(var)
                neighborhood_dict[param][0] = np.insert(neighborhood_dict[param][0], inc_at, incumbent_array[index])
                neighborhood_dict[param][1] = np.insert(neighborhood_dict[param][1], inc_at, self.incumbent[param])
                if not added_inc:
                    mean, var = self._predict_over_instance_set(impute_inactive_values(self.incumbent))
                    performance_dict[param].append(mean)
                    variance_dict[param].append(var)
            else:
                self.logger.info('Parameter is inactive')
        self.neighborhood_dict = neighborhood_dict
        self.performance_dict = performance_dict
        self.variance_dict = variance_dict

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
        for param in self.incumbent.keys():
            if param in self.performance_dict:
                fig = plt.figure(dpi=250)
                ax1 = fig.add_subplot(111)
                plt.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=.95)
                p, v = self.performance_dict[param], self.variance_dict[param]

                if not isinstance(self.incumbent.configuration_space.get_hyperparameter(param),
                                  CategoricalHyperparameter):
                    upper = np.array(list(map(lambda x, y: x + np.sqrt(y), p, v))).flatten()
                    if self.scenario.run_obj == "runtime":
                        lower = np.array(list(map(lambda x, y: max(x - np.sqrt(y), 0), p, v))).squeeze()
                    else:
                        lower = np.array(list(map(lambda x, y: x - np.sqrt(y), p, v))).squeeze()
                    ax1.fill_between(self.neighborhood_dict[param][1], lower, upper, label='std', color=self.area_color)
                    ax1.plot(self.neighborhood_dict[param][1], p, label='Predicted Performance', ls='-', zorder=80,
                             **self.LINE_FONT)
                    label = True
                    for n_idx, neighbor in enumerate(self.neighborhood_dict[param][1]):
                        if neighbor == self.incumbent[param]:
                            ax1.scatter(neighbor, p[n_idx], label='incumbent', c='r', marker='.', zorder=999)
                        else:
                            if label:
                                ax1.scatter(neighbor, p[n_idx], label='query points', c='w', marker='.',
                                            zorder=100, edgecolors='k')
                                label = False
                            else:
                                ax1.scatter(neighbor, p[n_idx], c='w', marker='.', zorder=100, edgecolors='k')
                    ax1.xaxis.grid(True)
                    ax1.yaxis.grid(True)
                else:
                    b = ax1.boxplot([[x] for x in p], showfliers=False)
                    plt.xticks(np.arange(1, len(self.neighborhood_dict[param][1])+1, 1),
                               self.neighborhood_dict[param][1])
                    min_y = min(p)
                    max_y = max(p)
                    std = np.array(list(map(lambda s: np.sqrt(s), v))).flatten()
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
                            min_y = min(min_y, max(y[0] - std_, -0.1))
                        else:
                            min_y = min(min_y, y[0] - std_)
                        max_y = max(max_y, y[2] + std_)
                    plt.ylim([min_y, max_y])
                    for idx, t in enumerate(ax1.xaxis.get_ticklabels()):
                        color_ = (0., 0., 0.)
                        if self.neighborhood_dict[param][1][idx] == self.incumbent[param]:
                            color_ = (1, 0, 0)
                        t.set_color(color_)

                plt.title(param)
                ax1.legend()
                if self.scenario.run_obj == 'runtime':
                    ax1.set_ylabel('runtime [sec]', zorder=81, **self.LABEL_FONT)
                else:
                    ax1.set_ylabel('%s' % self.scenario.run_obj, zorder=81, **self.LABEL_FONT)
                try:
                    plt.tight_layout()
                except ValueError:
                    pass
                plt.savefig(os.path.join(name, param + '.png'))
