# PYTHON_ARGCOMPLETE_OK
import os
if not 'MATPLOTLIBRC' in os.environ:
    os.environ['MATPLOTLIBRC'] = os.path.join(os.path.dirname(__file__))
import json
import datetime
import logging
import time
import warnings
from typing import List
from typing import Dict
from typing import Union

import numpy as np
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.tae.execute_ta_run_aclib import StatusType
from ConfigSpace.configuration_space import Configuration
from pimp.importance.importance import Importance
from pimp.utils.io.cmd_reader import CMDs

from smac.facade.smac_ac_facade import SMAC4AC as SMAC

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class PIMP:
    def __init__(self,
                 scenario: Scenario,
                 smac: Union[SMAC, None] = None,
                 mode: str = 'all',
                 X: Union[None, List[list], np.ndarray] = None,
                 y: Union[None, List[list], np.ndarray] = None,
                 numParams: int = -1,
                 impute: bool = False,
                 seed: int = 12345,
                 run: bool = False,
                 max_sample_size: int = -1,
                 fanova_cut_at_default: bool = False,
                 fANOVA_pairwise: bool = True,
                 forwardsel_feat_imp: bool = False,
                 incn_quant_var: bool = True,
                 marginalize_away_instances: bool = False,
                 save_folder: str = 'PIMP'):
        """
        Interface to be used with SMAC or with X and y matrices.
        :param scenario: The scenario object, that knows the configuration space.
        :param smac: The smac object that keeps all the run-data
        :param mode: The mode with which to run PIMP [ablation, fanova, all, forward-selection]
        :param X: Numpy Array that contains parameter arrays
        :param y: Numpy array that contains the corresponding performance values
        :param numParams: The number of parameters to evaluate
        :param impute: Flag to decide if censored data gets imputed or not
        :param seed: The random seed
        :param run: Flag to immediately compute the importance values after this setup or not.
        """
        self.scenario = scenario
        self.imp = None
        self.mode = mode
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder): os.mkdir(self.save_folder)
        if smac is not None:
            self.imp = Importance(scenario=scenario,
                                  runhistory=smac.runhistory,
                                  incumbent=smac.solver.incumbent,
                                  seed=seed,
                                  parameters_to_evaluate=numParams,
                                  save_folder='PIMP',
                                  impute_censored=impute,
                                  max_sample_size=max_sample_size,
                                  fANOVA_cut_at_default=fanova_cut_at_default,
                                  fANOVA_pairwise=fANOVA_pairwise,
                                  forwardsel_feat_imp=forwardsel_feat_imp,
                                  incn_quant_var=incn_quant_var,
                                  preprocess=marginalize_away_instances)
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
            runHist = RunHistory()
            if X.shape[0] != y.shape[0]:
                raise Exception('Number of samples in X and y dont match!')
            n_params = len(scenario.cs.get_hyperparameters())
            feats = None
            if X.shape[1] > n_params:
                feats = X[:, n_params:]
                assert feats.shape[1] == scenario.feature_array.shape[1]
                X = X[:, :n_params]

            for p in range(X.shape[1]):  # Normalize the data to fit into [0, 1]
                _min, _max = np.min(X[:, p]), np.max(X[:, p])
                if _min < 0. or 1 < _max:  # if it is not already normalized
                    for id, v in enumerate(X[:, p]):
                        X[id, p] = (v - _min) / (_max - _min)

            # Add everything to a runhistory such that PIMP can work with it
            for x, feat, y_val in zip(X, feats if feats is not None else X, y):
                id = None
                for inst in scenario.feature_dict:  # determine on which instance a configuration was run
                    if np.all(scenario.feature_dict[inst] == feat):
                        id = inst
                        break
                runHist.add(Configuration(scenario.cs, vector=x), y_val, 0, StatusType.SUCCESS, id)
            self.X = X
            self.y = y

            best_ = None  # Determine incumbent according to the best mean cost in the runhistory
            for config in runHist.config_ids:
                inst_seed_pairs = runHist.get_runs_for_config(config)
                all_ = []
                for inst, seed in inst_seed_pairs:
                    rk = RunKey(runHist.config_ids[config], inst, seed)
                    all_.append(runHist.data[rk].cost)
                mean = np.mean(all_)
                if best_ is None or best_[0] > mean:
                    best_ = (mean, config)
            incumbent = best_[1]
            self.imp = Importance(scenario=scenario,
                                  runhistory=runHist,
                                  seed=seed,
                                  parameters_to_evaluate=numParams,
                                  save_folder=self.save_folder,
                                  impute_censored=impute,
                                  incumbent=incumbent,
                                  fANOVA_cut_at_default=fanova_cut_at_default,
                                  fANOVA_pairwise=fANOVA_pairwise,
                                  forwardsel_feat_imp=forwardsel_feat_imp,
                                  incn_quant_var=incn_quant_var,
                                  preprocess=marginalize_away_instances
                                  )
        else:
            raise Exception('Neither X and y matrices nor a SMAC object were specified to compute the importance '
                            'values from!')

        if run:
            self.compute_importances()

    def compute_importances(self):
        if self.mode == 'all':
            self.mode = ['ablation',
                         'forward-selection',
                         'fanova',
                         'incneighbor']
        elif not isinstance(self.mode, list):
            self.mode = [self.mode]
        result = self.imp.evaluate_scenario(self.mode, save_folder=self.save_folder)
        return result

    def plot_results(self, result: Union[List[Dict[str, float]], Dict[str, float]], save_table: bool = True,
                     show=False):
        save_folder = self.save_folder
        if self.mode == 'all':
            with open(os.path.join(save_folder, 'pimp_values_%s.json' % self.mode), 'w') as out_file:
                json.dump(result[0], out_file, sort_keys=True, indent=4, separators=(',', ': '))
            self.imp.plot_results(list(map(lambda x: os.path.join(save_folder, x.name.lower()), result[1])),
                                  result[1], show=show)
            if save_table:
                self.imp.table_for_comparison(evaluators=result[1], name=os.path.join(
                    save_folder, 'pimp_table_%s.tex' % self.mode), style='latex')
            else:
                self.imp.table_for_comparison(evaluators=result[1], style='cmd')
        else:
            with open(os.path.join(save_folder, 'pimp_values_%s.json' % self.mode), 'w') as out_file:
                json.dump(result[0], out_file, sort_keys=True, indent=4, separators=(',', ': '))
            if isinstance(self.mode, list):
                self.imp.plot_results(name=os.path.join(save_folder, 'all'), show=show)
            else:
                self.imp.plot_results(name=os.path.join(save_folder, self.mode), show=show)


def cmd_line_call():
    """
    Main Parameter importance script.
    """
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()  # read cmd args
    cwd = os.path.abspath(os.getcwd())
    if args.out_folder and not os.path.isabs(args.out_folder):
        args.out_folder = os.path.abspath(args.out_folder)
    if args.trajectory and not os.path.isabs(args.trajectory):
        args.trajectory = os.path.abspath(args.trajectory)
    if not os.path.isabs(args.scenario_file):
        args.scenario_file = os.path.abspath(args.scenario_file)
    if not os.path.isabs(args.history):
        args.history = os.path.abspath(args.history)
    os.chdir(args.wdir)
    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    fanova_ready = True

    try:
        import fanova
    except ImportError:
        warnings.simplefilter('always', ImportWarning)
        warnings.warn('fANOVA is not installed in your environment. To install it please run '
                      '"git+http://github.com/automl/fanova.git@master"')
        fanova_ready = False

    if 'influence-model' in args.modus:
        logging.warning('influence-model not fully supported yet!')
    if 'incneighbor' in args.modus:
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('incneighbor will be deprecated in version 1.0.0 as it was the development name of'
                      ' lpi. Use lpi instead.', DeprecationWarning, stacklevel=2)
    if 'lpi' in args.modus:  # LPI will replace incneighbor in the future
        args.modus[args.modus.index('lpi')] = 'incneighbor'
    if 'fanova' in args.modus and not fanova_ready:
        raise ImportError('fANOVA is not installed! To install it please run '
                          '"git+http://github.com/automl/fanova.git@master"')
    if 'all' in args.modus:
        choices = ['ablation',
                   'forward-selection',
                   'fanova',
                   'incneighbor']
        if not fanova_ready:
            raise ImportError('fANOVA is not installed! To install it please run '
                              '"git+http://github.com/automl/fanova.git@master"')
        del args.modus[args.modus.index('all')]
        if len(args.modus) == len(choices):
            pass
        else:
            args.modus = choices
    if not args.out_folder:
        if len(args.modus) > 1:
            tmp = ['all']
        else:
            tmp = args.modus
            if 'incneighbor' in args.modus:
                tmp = ['lpi']
        save_folder = os.path.join(cwd, 'PIMP_%s' % '_'.join(tmp))
        if os.path.exists(os.path.abspath(save_folder)):
            save_folder = os.path.join(cwd, 'PIMP_%s_%s' % ('_'.join(tmp), ts))
    else:
        if len(args.modus) > 1:
            tmp = ['all']
        else:
            tmp = args.modus
            if 'incneighbor' in args.modus:
                tmp = ['lpi']
        if os.path.exists(os.path.abspath(args.out_folder)) or os.path.exists(os.path.abspath(
                        args.out_folder + '_%s' % '_'.join(tmp))):
            save_folder = os.path.join(cwd, args.out_folder + '_%s_%s' % ('_'.join(tmp), ts))
        else:
            save_folder = os.path.join(cwd, args.out_folder + '_%s' % '_'.join(tmp))

    importance = Importance(scenario_file=args.scenario_file,
                            runhistory_file=args.history,
                            parameters_to_evaluate=args.num_params,
                            traj_file=args.trajectory, seed=args.seed,
                            save_folder=save_folder,
                            impute_censored=args.impute,
                            max_sample_size=args.max_sample_size,
                            fANOVA_cut_at_default=args.fanova_cut_at_default,
                            fANOVA_pairwise=args.fanova_pairwise,
                            forwardsel_feat_imp=args.forwardsel_feat_imp,
                            incn_quant_var=args.incn_quant_var,
                            preprocess=args.marg_inst,
                            forwardsel_cv=args.forwardsel_cv)  # create importance object
    with open(os.path.join(save_folder, 'pimp_args.json'), 'w') as out_file:
        json.dump(args.__dict__, out_file, sort_keys=True, indent=4, separators=(',', ': '))
    result = importance.evaluate_scenario(args.modus, save_folder=save_folder)
    if args.table:
        importance.table_for_comparison(evaluators=result[1], name=os.path.join(
            save_folder, 'pimp_table_%s.tex' % args.modus), style='latex')
    else:
        importance.table_for_comparison(evaluators=result[1], style='cmd')
    os.chdir(cwd)


if __name__ == '__main__':
    cmd_line_call()
