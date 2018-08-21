# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import CamelAlgorithm
from NiaPy.util import TaskConvPrint, TaskConvPlot

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
			for i in range(D): val += sol[i] * sol[i]
			return val
		return evaluate

def simple_example(runs=10):
	for i in range(runs):
		Algorithm = CamelAlgorithm(NP=50, D=50, nGEN=50000, nFES=500000, omega=0.25, alpha=0.15, mu=0.5, S_init=1, E_init=1, T_min=0, T_max=100, benchmark=MyBenchmark())
		Best = Algorithm.run()
		logger.info('%s %s' % (Best[0], Best[1]))

def logging_example():
	task = TaskConvPrint(D=50, nFES=50000, nGEN=50000, benchmark=MyBenchmark())
	algo = CamelAlgorithm(NP=50,  omega=0.25, alpha=0.15, mu=0.5, S_init=1, E_init=1, T_min=0, T_max=100, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example():
	task = TaskConvPlot(D=50, nFES=50000, nGEN=10000, benchmark=MyBenchmark())
	algo = CamelAlgorithm(NP=50,  omega=0.25, alpha=0.15, mu=0.5, S_init=1, E_init=1, T_min=0, T_max=100, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

if __name__ == '__main__':
	if len(sys.argv) <= 1: simple_example(1)
	elif sys.argv[1] == 'plot': plot_example()
	elif sys.argv[1] == 'log': logging_example()
	else: simple_example(10)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
