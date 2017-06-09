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
                            save_folder=save_folder,
                            impute_censored=args.impute)  # create importance object
    save_folder += '_run1'
    with open(os.path.join(save_folder, 'pimp_args.json'), 'w') as out_file:
        json.dump(args.__dict__, out_file, sort_keys=True, indent=4, separators=(',', ': '))
    result = importance.evaluate_scenario(args.modus, sort_by=args.order)

    if args.modus == 'all':
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result[0], out_file, sort_keys=True, indent=4, separators=(',', ': '))
        importance.plot_results(list(map(lambda x: os.path.join(save_folder, x.name.lower()), result[1])),
                                result[1])
        if args.table:
            importance.table_for_comparison(evaluators=result[1], name=os.path.join(
                save_folder, 'pimp_table_%s.tex' % args.modus), style='latex')
        else:
            importance.table_for_comparison(evaluators=result[1], style='cmd')
    else:
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % args.modus), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))

        importance.plot_results(name=os.path.join(save_folder, args.modus))
