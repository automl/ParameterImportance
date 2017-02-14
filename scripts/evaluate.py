import logging
import sys
import os
import inspect
import datetime
import time
import json
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from pimp.importance.importance import Importance
from pimp.utils.io.cmd_reader import CMDs

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"

if __name__ == '__main__':
    """
    Main Parameter importance script.
    """
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()  # read cmd args
    logging.basicConfig(level=args.verbose_level)
    importance = Importance(args.scenario_file, args.history,
                            parameters_to_evaluate=args.num_params,
                            traj_file=args.trajectory, seed=args.seed)  # create importance object
    importance_value_dict = importance.evaluate_scenario(args.modus)

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    with open('pimp_values_%s_%s.json' % (args.modus, ts), 'w') as out_file:
        json.dump(importance_value_dict, out_file)

    importance.plot_results(name=args.modus)

