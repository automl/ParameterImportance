import datetime
import inspect
import json
import logging
import os
import sys
import time

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
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')
    save_folder = 'PIMP_%s_%s' % (args.modus, ts)

    importance = Importance(args.scenario_file, args.history,
                            parameters_to_evaluate=args.num_params,
                            traj_file=args.trajectory, seed=args.seed,
                            save_folder=save_folder)  # create importance object
    # print(args.__dict__)
    with open(os.path.join(save_folder, 'pimp_args.json'), 'w') as out_file:
        json.dump(args.__dict__, out_file)
    result = importance.evaluate_scenario(args.modus)

    if args.modus == 'all':
        tmp_res = result['evaluators']
        result['evaluators'] = None
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result, out_file)
        result['evaluators'] = tmp_res
        importance.plot_results(list(map(lambda x: os.path.join(save_folder, x), result['methods'])),
                                result['evaluators'])
    else:
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result, out_file)

        importance.plot_results(name=os.path.join(save_folder, args.modus))
