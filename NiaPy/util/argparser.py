# encoding=utf8

"""Arguments parser utility functions."""

import sys
import logging
from argparse import ArgumentParser
from numpy import inf
import NiaPy.benchmarks as bencs

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.argparse')
logger.setLevel('INFO')

__all__ = ['MakeArgParser', 'getArgs', 'getDictArgs']


def makeCbechs():
    return bencs.__all__


def optimizationType(x):
    r"""Map function for optimization type.

    Args:
        x (str): String representing optimization type.

    Returns:
        OptimizationType: Optimization type based on type that is defined as enum.

    """

    if x not in ['min', 'max']:
        logger.info('You can use only [min, max], using min')
    from NiaPy.task import OptimizationType
    return OptimizationType.MAXIMIZATION if x == 'max' else OptimizationType.MINIMIZATION


def MakeArgParser():
    r"""Create/Make parser for parsing string.

    Parser:
        * `-a` or `--algorithm` (str):
            Name of algorithm to use. Default value is `jDE`.
        * `-b` or `--bench` (str):
            Name of benchmark to use. Default values is `Benchmark`.
        * `-D` (int):
            Number of dimensions/components usd by benchmark. Default values is `10`.
        * `-nFES` (int):
            Number of maximum function evaluations. Default values is `inf`.
        * `-nGEN` (int):
            Number of maximum algorithm iterations/generations. Default values is `inf`.
        * `-NP` (int):
            Number of individuals in population. Default values is `43`.
        * `-r` or `--runType` (str);
            Run type of run. Value can be (Default value is `''`.):
                * '': No output during the run. Output is shown only at the end of algorithm run.
                * `log`: Output is shown every time new global best solution is found
                * `plot`: Output is shown only at the end of run. Output is shown as graph plotted in mathplotlib. Graph represents convegance of algorithm over run time of algorithm.
        * `-seed` (list of int or int):
            Set the starting seed of algorithm run. If multiple runs, user can provide list of ints, where each int usd use at new run. Default values is `None`.
        * `-optType` (str):
            Optimization type of the run. Values can be (Default value is `min`.):
                * `min`: For minimization problems
                * `max`: For maximization problems

    Returns:
        ArgumentParser: Parser for parsing arguments from string.

    See Also:
        * :class:`ArgumentParser`
        * :func:`ArgumentParser.add_argument`

    """

    parser, cbechs = ArgumentParser(description='Runer example.'), makeCbechs()
    parser.add_argument('-a', '--algorithm', dest='algo', default='jDE', type=str)
    parser.add_argument('-b', '--bench', dest='bench', nargs='*', default=cbechs[0], choices=cbechs, type=str)
    parser.add_argument('-D', dest='D', default=10, type=int)
    parser.add_argument('-nFES', dest='nFES', default=inf, type=int)
    parser.add_argument('-nGEN', dest='nGEN', default=inf, type=int)
    parser.add_argument('-NP', dest='NP', default=43, type=int)
    parser.add_argument('-r', '--runType', dest='runType', choices=['', 'log', 'plot'], default='', type=str)
    parser.add_argument('-seed', dest='seed', nargs='+', default=[None], type=int)
    parser.add_argument('-optType', dest='optType', default=optimizationType('min'), type=optimizationType)
    return parser


def getArgs(av):
    r"""Parse arguments form inputed string.

    Args:
        av (str): String to parse.

    Returns:
        Dict[str, Union[float, int, str, OptimizationType]]: Where key represents argument name and values it's value.

    See Also:
        * :func:`NiaPy.util.argparser.MakeArgParser`.
        * :func:`ArgumentParser.parse_args`

    """

    parser = MakeArgParser()
    a = parser.parse_args(av)
    return a


def getDictArgs(argv):
    r"""Pasre input string.

    Args:
        argv (str): Input string to parse for arguments

    Returns:
        dict: Parsed input string

    See Also:
        * :func:`NiaPy.utils.getArgs`

    """

    return vars(getArgs(argv))


if __name__ == '__main__':
    args = getArgs(sys.argv[1:])
    logger.info(str(args))
