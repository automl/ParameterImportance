import os
import json
import datetime
import logging
import time
from typing import List
from typing import Dict
from typing import Union

import numpy as np
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.optimizer.objective import average_cost
from smac.tae.execute_ta_run_aclib import StatusType
from ConfigSpace.configuration_space import Configuration

from pimp.importance.importance import Importance
from pimp.utils.io.cmd_reader import CMDs


__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class PIMP:
    def __init__(self, scenario: Scenario, smac: Union[SMAC, None]=None, mode: str='all',
                 X: Union[None, List[list], np.ndarray]=None, y: Union[None, List[list], np.ndarray]=None,
                 numParams: int=-1, impute: bool=False, seed: int=12345, run: bool=False):
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
        self.save_folder = scenario.output_dir
        if smac is not None:
            self.imp = Importance(scenario=scenario, runhistory=smac.runhistory, incumbent=smac.solver.incumbent,
                                  seed=seed, parameters_to_evaluate=numParams, save_folder='PIMP',
                                  impute_censored=impute)
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
            runHist = RunHistory(average_cost)
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
            self.imp = Importance(scenario=scenario, runhistory=runHist, seed=seed, parameters_to_evaluate=numParams,
                                  save_folder='PIMP', impute_censored=impute, incumbent=incumbent)
        else:
            raise Exception('Neither X and y matrices nor a SMAC object were specified to compute the importance '
                            'values from!')

        if run:
            return self.compute_importances()

    def compute_importances(self, order=3):
        result = self.imp.evaluate_scenario(self.mode, sort_by=order)
        return result

    def plot_results(self, result: Union[List[Dict[str, float]], Dict[str, float]], save_table: bool=True,
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
                json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))

            self.imp.plot_results(name=os.path.join(save_folder, self.mode), show=show)


def cmd_line_call():
    """
    Main Parameter importance script.
    """
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()  # read cmd args
    logging.basicConfig(level=args.verbose_level)
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    if not args.out_folder:
        save_folder = 'PIMP_%s_%s' % (args.modus, ts)
    else:
        if os.path.exists(os.path.abspath(args.out_folder)):
            save_folder = args.out_folder + '_%s_%s' % (args.modus, ts)
        else:
            save_folder = args.out_folder + '_%s' % args.modus

    importance = Importance(scenario_file=args.scenario_file, runhistory_file=args.history,
                            parameters_to_evaluate=args.num_params,
                            traj_file=args.trajectory, seed=args.seed,
                            save_folder=save_folder,
                            impute_censored=args.impute)  # create importance object
    with open(os.path.join(save_folder, 'pimp_args.json'), 'w') as out_file:
        json.dump(args.__dict__, out_file, sort_keys=True, indent=4, separators=(',', ': '))
    result = importance.evaluate_scenario(args.modus, sort_by=args.order)

    if args.modus == 'all':
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result[0], out_file, sort_keys=True, indent=4, separators=(',', ': '))
        importance.plot_results(list(map(lambda x: os.path.join(save_folder, x.name.lower()), result[1])),
                                result[1], show=False)
        if args.table:
            importance.table_for_comparison(evaluators=result[1], name=os.path.join(
                save_folder, 'pimp_table_%s.tex' % args.modus), style='latex')
        else:
            importance.table_for_comparison(evaluators=result[1], style='cmd')
    else:
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))

        importance.plot_results(name=os.path.join(save_folder, args.modus), show=False)


if __name__ == '__main__':
    cmd_line_call()
