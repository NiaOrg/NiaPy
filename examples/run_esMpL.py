# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import EvolutionStrategyMpL
from NiaPy.benchmarks.utility import TaskConvPrint, TaskConvPlot

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

class MyBenchmark(object):
	def __init__(self):
		self.Lower = -11
		self.Upper = 11

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate

def simple_example(runs=10):
	for i in range(runs):
		algo = EvolutionStrategyMpL(D=50, nFES=50000, seed=None, benchmark=MyBenchmark())
		best = algo.run()
		logger.info('%s %s' % (best[0], best[1]))

def logging_example():
	task = TaskConvPrint(D=10, nFES=50000, nGEN=50000, benchmark=MyBenchmark())
	algo = EvolutionStrategyMpL(mu=50, lam=40, k=60, c_a=1.5, c_r=0.25, seed=None, task=task)
	best = algo.run()
	logger.info('nFES:%s nGEN:%s\n%s %s' % (task.Evals, task.Iters, best[0], best[1]))

def plot_example():
	task = TaskConvPlot(D=50, nFES=50000, nGEN=10000, benchmark=MyBenchmark())
	algo = EvolutionStrategyMpL(mu=65, lam=50, k=25, c_a=1.5, c_r=0.25, seed=None, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

# simple_example()
logging_example()
# plot_example()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
