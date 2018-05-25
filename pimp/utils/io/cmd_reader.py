import logging
import argcomplete
import re as _re
import os
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser, HelpFormatter
from argparse import OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE, REMAINDER, PARSER


from pimp.__version__ import __version__ as v

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


############################ CUSTUMIZATION OF ARGPARSE OUTPUT #######################################
try:
    from gettext import gettext as _, ngettext
except ImportError:
    def _(message):
        return message

    def ngettext(singular,plural,n):
        if n == 1:
            return singular
        else:
            return plural


class SmartArgsDefHelpFormatter(ArgumentDefaultsHelpFormatter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_help_position = 40
        self._width = 120

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not SUPPRESS and action.nargs is None:
                help += '(default: %(default)s)'
        return help

    def _split_lines(self, text, width):
        if text.startswith('raw|'):
            return text[4:].splitlines()
        return HelpFormatter._split_lines(self, text, width)

    def _format_args(self, action, default_metavar):
        """
        Adaptation of argparse _format_args to ignore special output for variable nargs
        """
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '[%s]' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            result = '%s' % get_metavar(1)  # '[%s [%s ...]]' % get_metavar(2)
        elif action.nargs == ONE_OR_MORE:
            result = '%s' % get_metavar(1)  # '%s [%s ...]' % get_metavar(2)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s ...' % get_metavar(1)
        else:
            formats = ['%s' for _ in range(action.nargs)]
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choices = [action.choices[0], action.choices[-1]]
            choice_strs = [str(choice) for choice in choices]
            if len(action.choices) <= 2:
                result = '{%s}' % ', '.join(choice_strs)
            else:
                result = '{%s}' % '{}, ..., {}'.format(*choice_strs)
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format

    def _format_usage(self, usage, actions, groups, prefix):
        """
        Nearly identical to argparses _format usage except suppressing an assert to allow for custom choice behaviour
        """
        if prefix is None:
            prefix = _('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                part_regexp = r'\(.*?\)+|\[.*?\]+|\S+'
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = _re.findall(part_regexp, opt_usage)
                pos_parts = _re.findall(part_regexp, pos_usage)
                # assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)


##################################### ACTUAL PARSER SETUP ##############################
class CMDs:
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
        global v
        m_choices = ['ablation',
                     'all',
                     'fanova',
                     'forward-selection',
                     'incneighbor',
                     'influence-model',
                     'lpi']
        parser = ArgumentParser(formatter_class=SmartArgsDefHelpFormatter,
                                description='%s' % (
                                    "PIMP implements a multitude of "
                                    "parameter importance methods to determine which parameters have the most "
                                    "influence over the algorithms behaviour, given data from an algorithm configurator"
                                    "such as SMAC3. Version " + str(v)),
                                add_help=False)
        req_opts = parser.add_argument_group("required arguments:" + '~'*100)
        req_opts.add_argument("-S", "--scenario_file",
                              required=True,
                              help="(path to) scenario file in AClib format.",
                              default=SUPPRESS)
        req_opts.add_argument("-M", "--modus",
                              required=True,
                              nargs='+',
                              choices=m_choices,
                              help='analysis method(s) to use. Choose any combination from \n{' +
                                   ', '.join(m_choices) + '}',
                              type=str.lower,
                              default=SUPPRESS)
        req_opts.add_argument("-H", "--history",
                              required=True,
                              help="(path to) runhistory file that contains all "
                                   "collected data from an optimization run",
                              default=SUPPRESS)

        opt_opts = parser.add_argument_group("optional arguments:" + '~'*100)
        opt_opts.add_argument("--seed",
                              default=12345,
                              type=int,
                              help="raw|random seed. Used internally when fitting the random forest.\n")
        opt_opts.add_argument("-V", "--verbose_level",
                              default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="raw|verbosity\n")
        opt_opts.add_argument("-T", "--trajectory",
                              default=None,
                              help="raw|(path to) trajectory file. Needs to contain at least one line detailing the\n"
                                   "best seen configuration and it's cost.\n")
        opt_opts.add_argument("-N", "--num_params",
                              default=-1,
                              type=int,
                              help="raw|number of parameters to evaluate. -1 -> use all\n")
        opt_opts.add_argument("-P", "--max_sample_size",
                              default=-1,
                              type=int,
                              help="raw|number of samples from runhistorie(s) used. -1 -> use all\n")
        opt_opts.add_argument('-F', '--out-folder',
                              default=None,
                              help='raw|folder to store all results in\n',
                              dest='out_folder')
        opt_opts.add_argument('-D', '--working_dir',
                              default='.',
                              help='raw|working directory. Contains all necessary files such as scenario, (path to)\n'
                                   'runhistory, [(path to) trajectory].\n',
                              dest='wdir')
        opt_opts.add_argument("-I", "--impute",
                              action='store_true',
                              help="If set, censored data will be imputed.")
        opt_opts.add_argument("-C", "--table",
                              action='store_true',
                              help="=> a .tex file will be created containing all results in an easy to compare"
                                   "fashion. Parameters will be sorted by importance according to the first method"
                                   "in the table.")
        opt_opts.add_argument('--fanova_cut_at_default',
                              action='store_true',
                              help='cut fANOVA results at the default. This quantifies importance only in'
                                   ' terms of improvement over the'
                                   ' default.')
        opt_opts.add_argument('--fanova_no_pairs',
                              action='store_false',
                              help="=> fANOVA won't compute pairwise marginals",
                              dest='fanova_pairwise')
        opt_opts.add_argument('--lpi_quantify_perf_improvement',
                              action='store_false',
                              help="=> LPI computes importance as performance improvement",
                              dest='incn_quant_var')
        opt_opts.add_argument('--forward_sel_feat_imp',
                              action='store_true',
                              help="=> forward selection computes feature importance, not parameter importance",
                              dest='forwardsel_feat_imp')
        opt_opts.add_argument('--forwardsel_cv',
                              action='store_true',
                              help='=> forward selection errors computed via cross-validation instead of OOBs',
                              dest='forwardsel_cv')
        opt_opts.add_argument('--marginalize_over_instances',
                              action='store_true',
                              help='=> deactivate preprocessing step in which instances are marginalized '
                                   'away to speedup'
                                   ' ablation, forward-selection and LPI predictions',
                              dest='marg_inst')
        spe_opts = parser.add_argument_group("special arguments:" + '~'*100)
        spe_opts.add_argument('-v', '--version', action='version',
                              version='%(prog)s ' + str(v), help="show program's version number and exit.")
        spe_opts.add_argument("-h", "--help", action="help", help="show this help message and exit")
        argcomplete.autocomplete(parser)
        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    def _check_args(self, args_):
        """Checks command line arguments (e.g., whether all given files exist)
        Parameters
        ----------
        args_: parsed arguments
            Parsed command line arguments
        Raises
        ------
        ValueError
            in case of missing files or wrong configurations
        """

        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))
