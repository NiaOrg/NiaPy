# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, too-many-function-args, old-style-class
import sys
import logging
from argparse import ArgumentParser
import NiaPy as np

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.argparse')
logger.setLevel('INFO')

__all__ = ['MakeArgParser', 'getArgs', 'getDictArgs']

def makeCbechs(): return np.benchmarks.__all__

def optimizationType(x):
	if x not in ['min', 'max']: logger.info('You can use only [min, max], using min')
	return np.util.OptimizationType.MAXIMIZATION if x == 'max' else np.util.OptimizationType.MINIMIZATION

def MakeArgParser():
	parser, cbechs = ArgumentParser(description='Runer example.'), makeCbechs()
	parser.add_argument('-a', '--algorithm', dest='algo', default='jDE', type=str)
	parser.add_argument('-b', '--bech', dest='bech', nargs='*', default=cbechs[0], choices=cbechs, type=str)
	parser.add_argument('-D', dest='D', default=10, type=int)
	parser.add_argument('-nFES', dest='nFES', default=50000, type=int)
	parser.add_argument('-nGEN', dest='nGEN', default=5000, type=int)
	parser.add_argument('-NP', dest='NP', default=43, type=int)
	parser.add_argument('-r', '--runType', dest='runType', choices=['', 'log', 'plot'], default='', type=str)
	parser.add_argument('-seed', dest='seed', default=None, type=int)
	parser.add_argument('-optType', dest='optType', default=optimizationType('min'), type=optimizationType)
	return parser

def getArgs(argv):
	parser = MakeArgParser()
	args = parser.parse_args(argv)
	return args

def getDictArgs(argv): return vars(getArgs(argv))

if __name__ == '__main__':
	args = getArgs(sys.argv[1:])
	print (args)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
