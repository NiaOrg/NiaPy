# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging

from matplotlib import pyplot as plt

from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm
from NiaPy.benchmarks import Benchmark, Katsuura, Elliptic
from NiaPy.task import StoppingTask

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

class MyBenchmark(Benchmark):
	Name = ['MyBenchmark']

	def __init__(self):
		self.Lower = -11
		self.Upper = 11

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate

benc = MyBenchmark()
benc.plot3d()
plt.show()

benc = Katsuura(-1, 1)
benc.plot3d(0.06)
plt.show()

benc = Elliptic()
benc.plot3d(0.65)
plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
