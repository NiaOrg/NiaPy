# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import FishSchoolSearch


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

# Common variables
school_size = 10
n_iter = 10
min_w = 1
w_scale = n_iter / 2.0
D = 10

SI_init = 0.1
SI_final = 0.01
SV_init = 10
SV_final = 0.1

logger.info('Running with custom MyBenchmark...')
for i in range(10):
    Algorithm = FishSchoolSearch(
        n_iter=10, school_size=10, D=10, SI_init=0.1, SI_final=0.001, SV_init=1, SV_final=0.1, min_w=1, w_scale=5, benchmark=MyBenchmark()
    )
    Best = Algorithm.run()
    logger.info(Best) 
