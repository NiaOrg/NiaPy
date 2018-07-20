# encoding=utf8
# pylint: disable=mixed-indentation
from unittest import TestCase
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm

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
		self.ca_custom = BareBonesFireworksAlgorithm(D=40, nFES=100000, n=10, C_a=2, C_r=0.5, MyBenchmark())
		self.ca_griewank = BareBonesFireworksAlgorithm(D=40, nFES=100000, n=10, C_a=5, C_r=0.5, 'griewank')

	def test_custom_works_fine(self): self.assertTrue(self.pso_custom.run())

	def test_griewank_works_fine(self): self.assertTrue(self.pso_griewank.run())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
