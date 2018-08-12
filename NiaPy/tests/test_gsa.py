# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from numpy import asarray
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import GravitationalSearchAlgorithm

class MyBenchmark:
	def __init__(self):
		self.Lower = -11
		self.Upper = 11

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class GSATestCase(TestCase):
	def setUp(self):
		self.gsa_custom = GravitationalSearchAlgorithm(D=10, NP=40, nFES=1000, benchmark=MyBenchmark())
		self.gsa_griewank = GravitationalSearchAlgorithm(NP=10, D=40, nFES=1000, benchmark='griewank')

	def test_Custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.gsa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.gsa_custom.task.D, asarray(x[0])), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.gsa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.gsa_griewank.task.D, asarray(x[0])), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
