# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from margparser import getArgs
from NiaPy.algorithms.basic import GravitationalSearchAlgorithm
from NiaPy.benchmarks.utility import TaskConvPrint, TaskConvPlot, OptimizationType

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

class MinMB(object):
	def __init__(self):
		self.Lower = -5.12
		self.Upper = 5.12

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class MaxMB(MinMB):
	def function(self):
		f = MinMB.function(self)
		def e(D, sol): return -f(D, sol)
		return e

def simple_example(runs=10, D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB):
	for i in range(runs):
		algo = GravitationalSearchAlgorithm(D=D, NP=40, nFES=nFES, F=0.5, CR=0.9, seed=seed + 1 if seed != None else seed, optType=optType, benchmark=optFunc())
		Best = algo.run()
		logger.info('%s %s' % (Best[0], Best[1]))

def logging_example(D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB):
	task = TaskConvPrint(D=D, nFES=nFES, nGEN=50000, optType=optType, benchmark=optFunc())
	algo = GravitationalSearchAlgorithm(NP=40, F=0.5, CR=0.9, seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example(D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB):
	task = TaskConvPlot(D=D, nFES=nFES, nGEN=10000, optType=optType, benchmark=optFunc())
	algo = GravitationalSearchAlgorithm(NP=40, F=0.5, CR=0.9, seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

def getOptType(strtype):
	if strtype == 'min': return OptimizationType.MINIMIZATION, MinMB
	elif strtype == 'max': return OptimizationType.MAXIMIZATION, MaxMB
	else: return None

if __name__ == '__main__':
	pargs = getArgs(sys.argv[1:])
	optType, optFunc = getOptType(pargs.optType)
	if not pargs.runType: simple_example(D=pargs.D, nFES=pargs.nFES, seed=pargs.seed, optType=optType, optFunc=optFunc)
	elif pargs.runType == 'log': logging_example(D=pargs.D, nFES=pargs.nFES, seed=pargs.seed, optType=optType, optFunc=optFunc)
	elif pargs.runType == 'plot': plot_example(D=pargs.D, nFES=pargs.nFES, seed=pargs.seed, optType=optType, optFunc=optFunc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
