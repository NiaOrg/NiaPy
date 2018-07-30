# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.benchmarks.utility import TaskConvPrint, TaskConvPlot
from NiaPy.algorithms.basic.ga import MutationUros, CrossoverUros

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
	for i in range(10):
		algo = GeneticAlgorithm(D=10, NP=40, nFES=100000, Ts=5, Mr=0.5, Cr=0.4, benchmark=MyBenchmark())
		Best = algo.run()
		logger.info('%s %s' % (Best[0], Best[1]))

def logging_example():
	task = TaskConvPrint(D=10, nFES=50000, nGEN=50000, benchmark=MyBenchmark())
	algo = GeneticAlgorithm(NP=40, Ts=4, Mr=0.2, Cr=0.5, Mutation=MutationUros, Crossover=CrossoverUros, seed=None, task=task)
	# algo = GeneticAlgorithm(NP=50, Ts=10, Mr=0.5, Cr=0.5, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example():
	task = TaskConvPlot(D=50, nFES=50000, nGEN=10000, benchmark=MyBenchmark())
	algo = GeneticAlgorithm(NP=40, Ts=5, Mr=0.5, Cr=0.4, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

# simple_example()
logging_example()
# plot_example()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
