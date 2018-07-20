# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm

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

for i in range(10):
	Algorithm = BareBonesFireworksAlgorithm(D=50, nFES=500000, n=15, C_a=2, C_r=0.4, benchmark=MyBenchmark())
	Best = Algorithm.run()
	logger.info('%s %s' % (Best[0], Best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
