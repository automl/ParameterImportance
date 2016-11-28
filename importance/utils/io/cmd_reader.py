import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from smac.utils.io.cmd_reader import CMDReader


class CMDs(CMDReader):  #

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
                              help=SUPPRESS, choices=['ablation', 'forward-selection', 'fANOVA'])
        req_opts.add_argument("--history", required=True,
                              help="runhistory file")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=12345, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbosity")

        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc
