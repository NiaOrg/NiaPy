# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

class MyBenchmark(object):
	def __init__(self):
		self.Lower = -5
		self.Upper = 5

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

for i in range(10):
	Algorithm = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=10000, benchmark=MyBenchmark())
	Best = Algorithm.run()
	logger.info(Best)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
