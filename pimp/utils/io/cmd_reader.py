import logging
import argcomplete
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
        req_opts.add_argument("-S", "--scenario_file",
                              required=True,
                              help="scenario file in AClib format")
        req_opts.add_argument("-M", "--modus",
                              required=True,
                              nargs='+',
                              help='Analysis method(s) to use',
                              choices=['ablation',
                                       'forward-selection',
                                       'influence-model',
                                       'all',
                                       'fanova',
                                       'lpi',
                                       'incneighbor'],
                              type=str.lower)
        req_opts.add_argument("-H", "--history",
                              required=True,
                              help="runhistory file")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed",
                              default=12345,
                              type=int,
                              help="random seed")
        req_opts.add_argument("-V", "--verbose_level",
                              default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbosity")
        req_opts.add_argument("-T", "--trajectory",
                              default=None,
                              help="Path to trajectory file")
        req_opts.add_argument("-N", "--num_params",
                              default=0,
                              type=int,
                              help="Number of parameters to evaluate")
        req_opts.add_argument("-P", "--max_sample_size",
                              default=-1,
                              type=int,
                              help="Number of samples from runhistorie(s) used. -1 -> use all")
        req_opts.add_argument("-I", "--impute",
                              action='store_true',
                              help="Impute censored data")
        req_opts.add_argument("-C", "--table",
                              action='store_true',
                              help="Save result table")
        req_opts.add_argument('-F', '--out-folder',
                              default=None,
                              help='Folder to store results in',
                              dest='out_folder')
        req_opts.add_argument('-D', '--working_dir',
                              default='.',
                              help='Directory to load all folders from.',
                              dest='wdir')
        req_opts.add_argument('--fanova_cut_at_default',
                              action='store_true',
                              help='Cut fANOVA results at the default. This quantifies importance only in'
                                   ' terms of improvement over the'
                                   ' default.')
        req_opts.add_argument('--fanova_no_pairs',
                              action='store_false',
                              help="fANOVA won't compute pairwise marginals",
                              dest='fanova_pairwise')
        req_opts.add_argument('--incneigh_quantify_perf_improvement',
                              action='store_false',
                              help="incumbent neighborhood computes importance via performance improvement",
                              dest='incn_quant_var')
        req_opts.add_argument('--forward_sel_feat_imp',
                              action='store_true',
                              help="forward selection for feature importance",
                              dest='forwardsel_feat_imp')
        req_opts.add_argument('--marginalize_over_instances',
                              action='store_true',
                              help='Deactivate preprocessing step in which instances are marginalized away to speedup'
                              ' ablation, forward-selection and incumbent neighborhood predictions',
                              dest='marg_inst')
        argcomplete.autocomplete(parser)
        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc
