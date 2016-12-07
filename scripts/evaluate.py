__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"

import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)

from importance.importance.importance import Importance
from importance.utils.io.cmd_reader import CMDs
import logging

if __name__ == '__main__':
    """
    Main Parameter importance script.
    """
    cmd_reader = CMDs()
    args, misc_ = cmd_reader.read_cmd()  # read cmd args
    logging.basicConfig(level=args.verbose_level)
    importance = Importance(args.scenario_file, args.history,
                            parameters_to_evaluate=args.num_params,
                            traj_file=args.trajectory)  # create importance object
    importance_value_dict = importance.evaluate_scenario(args.modus)
    importance.plot_results(name=args.modus)
