# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
import matplotlib.pyplot as plt
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.benchmarks.utility import TaskConvPrint, TaskConvPlot, OptimizationType
from margparser import getDictArgs

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

global_vector = []

class MinMB(object):
	def __init__(self):
		self.Lower = -11
		self.Upper = 11

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			global_vector.append(val)
			return val
		return evaluate

class MaxMB(MinMB):
	def function(self):
		f = MinMB.function(self)
		def e(D, sol): return -f(D, sol)
		return e

def run_defult():
	for i in range(10):
		Algorithm = FireflyAlgorithm(D=10, NP=20, nFES=1000, alpha=0.5, betamin=0.2, gamma=1.0, benchmark=MyBenchmark())	
		Best = Algorithm.run()
		plt.plot(global_vector)
		global_vector = []
		logger.info(Best)
	plt.xlabel('Number of evaluations')
	plt.ylabel('Fitness function value')
	plt.title('Convergence plot')
	plt.show()

def simple_example(runs=10, D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **no):
	for i in range(10):
		algo = FireflyAlgorithm(D=D, NP=20, nFES=nFES, alpha=0.5, betamin=0.2, gamma=1.0, seed=seed, optType=optType, benchmark=optFunc())
		Best = algo.run()
		logger.info('%s %s' % (Best[0], Best[1]))

def logging_example(D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **no):
	task = TaskConvPrint(D=D, nFES=nFES, nGEN=50000, optType=optType, benchmark=optFunc())
	algo = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example(D=10, nFES=50000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **no):
	task = TaskConvPlot(D=D, nFES=nFES, nGEN=10000, optType=optType, benchmark=optFunc())
	algo = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

def getOptType(strtype):
	if strtype == 'min': return OptimizationType.MINIMIZATION, MinMB
	elif strtype == 'max': return OptimizationType.MAXIMIZATION, MaxMB
	else: return None

if __name__ == '__main__':
	pargs = getDictArgs(sys.argv[1:])
	optType, optFunc = getOptType(pargs.pop('optType', 'min'))
	if not pargs['runType']: simple_example(optType=optType, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'log': logging_example(optType=optType, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'plot': plot_example(optType=optType, optFunc=optFunc, **pargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
