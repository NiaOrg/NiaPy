from unittest import TestCase
# pylint: disable=mixed-indentation
from NiaPy.algorithms.basic import CamelAlgorithm

class MyBenchmark(object):
	def __init__(self):
		self.Lower = -5.12
		self.Upper = 5.12

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class CSTestCase(TestCase):
	def setUp(self):
		# TODO set parameters
		self.ca_custom = CamelAlgorithm(40, 10000, , MyBenchmark())
		self.ca_griewank = CamelAlgorithm(40, 10000, , 'griewank')

	def test_custom_works_fine(self): self.assertTrue(self.pso_custom.run())

	def test_griewank_works_fine(self): self.assertTrue(self.pso_griewank.run())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
