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
from NiaPy.util import Task, TaskConvPrint, TaskConvPlot, OptimizationType, getDictArgs

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

def simple_example(alg, runs=10, D=10, nFES=50000, nGEN=10000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
	for i in range(runs):
		task = Task(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
		algo = alg(seed=seed, task=task)
		best = algo.run()
		logger.info('%s %s' % (best[0], best[1]))

def logging_example(alg, D=10, nFES=50000, nGEN=100000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
	task = TaskConvPrint(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
	algo = alg(seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example(alg, D=10, nFES=50000, nGEN=100000, seed=None, optType=OptimizationType.MINIMIZATION, optFunc=MinMB, **kn):
	task = TaskConvPlot(D=D, nFES=nFES, nGEN=nGEN, optType=optType, benchmark=optFunc())
	algo = alg(seed=seed, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

def getOptType(otype):
	if otype == OptimizationType.MINIMIZATION: return MinMB
	elif otype == OptimizationType.MAXIMIZATION: return MaxMB
	else: return None

if __name__ == '__main__':
	pargs, algo = getDictArgs(sys.argv[1:]), FireflyAlgorithm
	optFunc = getOptType(pargs['optType'])
	if not pargs['runType']: simple_example(algo, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'log': logging_example(algo, optFunc=optFunc, **pargs)
	elif pargs['runType'] == 'plot': plot_example(algo, optFunc=optFunc, **pargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
