import logging
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser

from smac.utils.io.cmd_reader import CMDReader

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class CMDs(CMDReader):

    """
        use argparse to parse command line options

        Attributes
        ----------
        logger : Logger oject
    """

    def read_cmd(self):
        """
            reads command line options

            Returns
            -------
                args_: parsed arguments; return of parse_args of ArgumentParser
        """

        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")
        req_opts.add_argument("--modus", required=True,
                              help='Analysis method to use', choices=['ablation', 'forward-selection',
                                                                      'influence-model', 'all', 'fanova'])
        req_opts.add_argument("--history", required=True,
                              help="runhistory file")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=12345, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbosity")
        req_opts.add_argument("--trajectory", default=None,
                              help="Path to trajectory file")
        req_opts.add_argument("--num_params", default=0, type=int,
                              help="Number of parameters to evaluate")

        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc
