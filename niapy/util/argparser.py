# encoding=utf8

"""Argparser class."""

import logging
import sys
from argparse import ArgumentParser

import numpy as np

import niapy.problems as problems
from niapy.task import OptimizationType

logging.basicConfig()
logger = logging.getLogger('niapy.util.argparse')
logger.setLevel('INFO')

__all__ = ['get_argparser', 'get_args', 'get_args_dict']


def _get_problem_names():
    r"""Get problem names."""
    return problems.__all__


def _optimization_type(x):
    r"""Get OptimizationType from string.

    Args:
        x (str): String representing optimization type.

    Returns:
        OptimizationType: Optimization type based on type that is defined as enum.

    """
    if x not in ['min', 'max']:
        logger.info('You can use only [min, max], using min')
    return OptimizationType.MAXIMIZATION if x == 'max' else OptimizationType.MINIMIZATION


def get_argparser():
    r"""Create/Make parser for parsing string.

    Parser:
        * `-a` or `--algorithm` (str):
            Name of algorithm to use. Default value is `jDE`.
        * `-p` or `--problem` (str):
            Name of problem to use. Default values is `Ackley`.
        * `-d` or `--dimension` (int):
            Number of dimensions/components used by problem. Default values is `10`.
        * `--max-evals` (int):
            Number of maximum function evaluations. Default values is `inf`.
        * `--max-iters` (int):
            Number of maximum algorithm iterations/generations. Default values is `inf`.
        * `-n` or  `--population-size` (int):
            Number of individuals in population. Default values is `43`.
        * `-r` or `--run-type` (str);
            Run type of run. Value can be:
                * '': No output during the run. Output is shown only at the end of algorithm run.
                * `log`: Output is shown every time new global best solution is found
                * `plot`: Output is shown only at the end of run. Output is shown as graph plotted in matplotlib. Graph represents convergence of algorithm over run time of algorithm.

            Default value is `''`.
        * `--seed` (list of int or int):
            Set the starting seed of algorithm run. If multiple runs, user can provide list of ints, where each int usd use at new run. Default values is `None`.
        * `--opt-type` (str):
            Optimization type of the run. Values can be:
                * `min`: For minimization problems
                * `max`: For maximization problems

            Default value is `min`.

    Returns:
        ArgumentParser: Parser for parsing arguments from string.

    See Also:
        * :class:`ArgumentParser`
        * :func:`ArgumentParser.add_argument`

    """
    parser, problem_names = ArgumentParser(description='Runner example.'), _get_problem_names()
    parser.add_argument('-a', '--algorithm', dest='algo', default='jDE', type=str)
    parser.add_argument('-p', '--problem', dest='problem', nargs='*', default=problem_names[0], choices=problem_names, type=str)
    parser.add_argument('-d', '--dimension', dest='dimension', default=10, type=int)
    parser.add_argument('--max-evals', dest='max_evals', default=np.inf, type=int)
    parser.add_argument('--max-iters', dest='max_iters', default=np.inf, type=int)
    parser.add_argument('-n', '--population-size', dest='population_size', default=43, type=int)
    parser.add_argument('-r', '--run-type', dest='run_type', choices=['', 'log', 'plot'], default='', type=str)
    parser.add_argument('--seed', dest='seed', nargs='+', default=[None], type=int)
    parser.add_argument('--opt-type', dest='opt_type', default=_optimization_type('min'), type=_optimization_type)
    return parser


def get_args(argv):
    r"""Parse arguments form input string.

    Args:
        argv (List[str]): List to parse.

    Returns:
        Dict[str, Union[float, int, str, OptimizationType]]: Where key represents argument name and values it's value.

    See Also:
        * :func:`niapy.util.argparser.get_argparser`.
        * :func:`ArgumentParser.parse_args`

    """
    parser = get_argparser()
    a = parser.parse_args(argv)
    return a


def get_args_dict(argv):
    r"""Parse input string.

    Args:
        argv (List[str]): Input string to parse for arguments

    Returns:
        dict: Parsed input string

    See Also:
        * :func:`niapy.utils.get_args`

    """
    return vars(get_args(argv))


if __name__ == '__main__':
    r"""Run the algorithms based on parameters from the command line interface."""
    args = get_args(sys.argv[1:])
    logger.info(str(args))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
