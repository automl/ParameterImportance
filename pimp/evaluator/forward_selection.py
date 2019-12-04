import time
from collections import OrderedDict

import matplotlib as mpl
import numpy as np
from bokeh.models import ColumnDataSource, CDSView, Span, BasicTickFormatter, Row, IndexFilter
from bokeh.plotting import figure

from pimp.utils.bokeh_helpers import shorten_unique, save_and_show

mpl.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from sklearn.model_selection import KFold
from sklearn.metrics.regression import mean_squared_error
from pimp.evaluator.base_evaluator import AbstractEvaluator

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class ForwardSelector(AbstractEvaluator):

    def __init__(self, scenario, cs, model, to_evaluate: int, rng, feature_imp: bool=False,
                 cv: bool=False, **kwargs):
        """
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
        """
        super().__init__(scenario, cs, model, to_evaluate, rng, **kwargs)
        self.name = 'Forward-Selection'
        self.logger = 'pimp.' + self.name
        self.feature_importance = feature_imp
        self.cv = cv
        self.kf = None
        self.logger.info('%snalyzing feature importance' % ('A' if feature_imp else 'Not a'))

    def _get_error(self, used, used_bounds):
        if self.cv:
            cv_errors = []
            for train_idx, test_idx, in self.kf.split(self.X):
                self._refit_model(self.types[used], self.bounds[used_bounds],
                                  self.X[train_idx.reshape(-1, 1), used],
                                  self.y[train_idx])  # refit the model every round
                pred = self.model.predict(self.X[test_idx.reshape(-1, 1), used])[0]
                cv_errors.append(np.sqrt(mean_squared_error(self.y[test_idx], pred)))
            cve = np.mean(cv_errors)
        else:
            cve = np.float('inf')
        self._refit_model(self.types[used], self.bounds[used_bounds],
                          self.X[:, used],
                          self.y)  # refit the model every round
        oob = self.model.rf.out_of_bag_error()
        return oob, cve

    def run(self) -> OrderedDict:
        """
        Implementation of the forward selection loop.
        Uses SMACs EPM (RF) wrt the configuration space to minimize the OOB error.
        :return:
        OrderedDict
            dict_keys (first key -> most important) -> OOB error
        """
        params = self.cs.get_hyperparameters()
        param_ids = list(range(len(params)))
        feature_ids = list(range(len(params), len(self.types)))
        used = []
        used_bounds = []
        if self.feature_importance:
            used.extend(range(0, len(self.bounds)))
            used_bounds.extend(range(0, len(self.bounds)))
            # names = list(map(lambda x: 'Feat #' + str(len(feature_ids) + x - len(self.types)), feature_ids))
            names = self.scenario.feature_names
            ids = feature_ids
            if self._to_eval <= 0:
                self.to_evaluate = len(feature_ids)
            else:
                self.to_evaluate = min(self._to_eval, len(ids))
        else:
            used.extend(range(len(params), len(self.model.types)))  # we don't want to evaluate the feature importance
            names = params
            ids = param_ids

        self.kf = KFold(n_splits=5)
        last_error = np.inf

        pbar = tqdm(range(self.to_evaluate), ascii=True,
                    desc='{: >.30s}: {: >7.4f} ({:s})'.format('None', np.inf, 'CV-RMSE' if self.cv else 'OOB'),
                    disable=not self.verbose)
        for round_ in pbar:  # Main Loop
            errors = []
            innerpbar = trange(len(names) + 1, ascii=True, desc='{:<40s}'.format(' '), leave=False, position=-1,
                               disable=not self.verbose)
            for idx, name in zip(ids, names):
                innerpbar.set_description('{:<40s}'.format(name if self.feature_importance else name.name))
                self.logger.debug('Evaluating %s' % name)
                used.append(idx)
                if not self.feature_importance:
                    used_bounds.append(idx)
                # self.logger.debug('Used parameters: %s' % str(used))
                # self.logger.debug('Used bounds of parameters: %s' % str(used_bounds))

                start = time.time()
                oob, cve = self._get_error(sorted(used), sorted(used_bounds))
                errors.append(cve if self.cv else oob)
                used.pop()
                if not self.feature_importance:
                    used_bounds.pop()
                self.logger.debug('Refitted RF (sec %.2f; CV-RMSE: %.4f, OOB: %.4f)' % (time.time() - start, cve, oob))
                innerpbar.update(1)
            else:  # Don't change the used/used_bounds -> add a none round
                innerpbar.set_description('{:<40s}'.format('None'))
                self.logger.debug('Evaluating None')
                start = time.time()
                oob, cve = self._get_error(sorted(used), sorted(used_bounds))
                errors.append(cve if self.cv else oob)
                self.logger.debug('Refitted RF (sec %.2f; CV-RMSE: %.4f, OOB: %.4f)' % (time.time() - start, cve, oob))
                if round_ == 0:  # Always keep track of the first None!
                    self.evaluated_parameter_importance['None'] = errors[-1]
                innerpbar.update(1)
            best_idx = np.argmin(errors)  # type: int
            lowest_error = errors[best_idx]

            if (best_idx == len(errors) - 1 or lowest_error >= last_error):
                # None was best, i.e. adding features did not improve
                pbar.set_description('{: >.30s}: {: >7.4f} ({:s})'.format('None', lowest_error,
                                                                          'CV-RMSE' if self.cv else 'OOB'))
                self.logger.info('Best result if None was added -> Early stopping')
                break
            else:
                last_error = lowest_error
                best_parameter = names.pop(best_idx)
                used.append(ids.pop(best_idx))
            if not self.feature_importance:
                used_bounds.append(used[-1])

            if self.feature_importance:
                self.logger.debug('%s: %.4f' % (best_parameter, lowest_error))
                self.evaluated_parameter_importance[best_parameter] = lowest_error
                pbar.set_description('{: >.30s}: {: >7.4f} ({:s})'.format(best_parameter, lowest_error,
                                                                          'CV-RMSE' if self.cv else 'OOB'))
            else:
                pbar.set_description('{: >.30s}: {: >7.4f} ({:s})'.format(best_parameter.name, lowest_error,
                                                                          'CV-RMSE' if self.cv else 'OOB'))
                self.evaluated_parameter_importance[best_parameter.name] = lowest_error
        all_res = {'imp': self.evaluated_parameter_importance, 'order': list(self.evaluated_parameter_importance.keys())}
        return all_res

    def _plot_result(self, name, bar=True, show=True):
        """
            plot oob score as bar charts
            Parameters
            ----------
            name
                file name to save plot
        """

        fig, ax = plt.subplots()
        params = list(self.evaluated_parameter_importance.keys())
        errors = list(self.evaluated_parameter_importance.values())
        max_to_plot = min(len(errors), self.MAX_PARAMS_TO_PLOT)

        ind = np.arange(len(errors))
        if bar:
            ax.bar(ind, errors, color=self.area_color)
        else:
            ax.plot(ind, errors, **self.LINE_FONT)

        ax.set_ylabel('CV-RMSE' if self.cv else 'OOB', **self.LABEL_FONT)
        if bar:
            ax.set_xticks(ind)
            ax.set_xlim(-.5, max_to_plot - 0.5)
        else:
            ax.set_xticks(ind)
            ax.set_xlim(0, max_to_plot - 1)
        ax.set_xticklabels(params, rotation=30, ha='right', **self.AXIS_FONT)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

        plt.tight_layout()
        if name is not None:
            fig.savefig(name)
            if show:
                plt.show()
        else:
            plt.show()

    def plot_result(self, name=None, show=True):
        self._plot_result(name + '-barplot.png', True, show)
        self._plot_result(name + '-chng.png', False, show)
        plt.close('all')
        self.logger.info('Saved plot as %s-[barplot|chng].png' % name)

    def plot_bokeh(self, plot_name=None, show_plot=False, barplot=True, chng=True):
        """
        Plot OOB-scores as bar- or linechart in bokeh.

        Parameters
        ----------
        plot_name: str
            path where to store the plot, None to not save it
        show_plot: bool
            whether or not to open plot in standard browser
        barplot: bool
            plot OOB as barchart
        chng: bool
            plot OOB as linechart

        Returns
        -------
        layout: bokeh.models.Row
            bokeh plot (can be used in notebook or comparted with components)
        """
        # Get all relevant data-points
        params = list(self.evaluated_parameter_importance.keys())
        p_names_short = shorten_unique(params)
        errors = list(self.evaluated_parameter_importance.values())
        max_to_plot = min(len(errors), self.MAX_PARAMS_TO_PLOT)
        plot_indices = sorted(range(len(errors)), key=lambda x: errors[x], reverse=True)[:max_to_plot]

        # Customizing plot-style
        bar_width = 25

        # Create ColumnDataSource for both plots
        source = ColumnDataSource(data=dict(parameter_names_short=p_names_short,
                                            parameter_names=params,
                                            parameter_importance=errors,
                                            ))
        plots = []

        view = CDSView(source=source, filters=[IndexFilter(plot_indices)])
        tooltips = [("Parameter", "@parameter_names"),
                    ("Importance", "@parameter_importance"),
                    ]
        if barplot:
            p = figure(x_range=p_names_short,
                       plot_height=350, plot_width=100 + max_to_plot * bar_width,
                       toolbar_location=None, tools="hover", tooltips=tooltips)

            p.vbar(x='parameter_names_short', top='parameter_importance', width=0.9, source=source, view=view)
            for value in [self.IMPORTANCE_THRESHOLD, -self.IMPORTANCE_THRESHOLD]:
                p.add_layout(Span(location=value, dimension='width', line_color='red', line_dash='dashed'))

            plots.append(p)

        if chng:
            p = figure(x_range=p_names_short,
                       plot_height=350, plot_width=100 + max_to_plot * bar_width,
                       toolbar_location=None, tools="hover", tooltips=tooltips)

            p.line(x='parameter_names_short', y='parameter_importance', source=source, view=view)

            plots.append(p)

        # Common styling:
        for p in plots:
            p.xaxis.major_label_orientation = 1.3
            p.xaxis.major_label_text_font_size = "14pt"
            p.yaxis.formatter = BasicTickFormatter(use_scientific=False)
            p.yaxis.axis_label = 'CV-RMSE' if self.cv else 'OOB'

        layout = Row(*plots)

        # Save and show...
        save_and_show(plot_name, show_plot, layout)

        return layout
