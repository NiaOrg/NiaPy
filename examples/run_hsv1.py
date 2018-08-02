# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import HarmonySearchV1
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
		algo = HarmonySearchV1(D=50, nFES=50000, HMS=50, r_accept=0.7, r_pa=0.2, benchmark=MyBenchmark())
		best = algo.run()
		logger.info('%s %s' % (best[0], best[1]))

def logging_example():
	task = TaskConvPrint(D=50, nFES=50000, nGEN=50000, benchmark=MyBenchmark())
	algo = HarmonySearchV1(HMS=50, r_accept=0.7, r_pa=0.2, bw_min=0.32, bw_max=1.5, seed=None, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))

def plot_example():
	task = TaskConvPlot(D=50, nFES=50000, nGEN=50000, benchmark=MyBenchmark())
	algo = HarmonySearchV1(HMS=50, r_accept=0.7, r_pa=0.2, b_range=1.1, task=task)
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	input('Press [enter] to continue')

# simple_example(1)
logging_example()
# plot_example()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
