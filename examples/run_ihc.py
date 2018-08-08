# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import HillClimbAlgorithm

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(4321)

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

for i in range(10):
	algo = HillClimbAlgorithm(D=50, nFES=500000, delta=0.85, benchmark=MyBenchmark())
	best = algo.run()
	logger.info('%s %s' % (best[0], best[1]))
	# dodatek
	# import numpy as np
	# logger.info('%s' % (MyBenchmark().function()(50, best[0])))
	# print (best[2])
	# logger.info('%s' % np.apply_along_axis(lambda x: np.sum(x ** 2), 1, best[2]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
