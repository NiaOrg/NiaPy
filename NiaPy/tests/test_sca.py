# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from numpy import array_equal
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import SineCosineAlgorithm

class MyBenchmark:
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

class BBFWATestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.sca_custom = SineCosineAlgorithm(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=1)
		self.sca_customc = SineCosineAlgorithm(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=1)
		self.sca_griewank = SineCosineAlgorithm(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=1)
		self.sca_griewankc = SineCosineAlgorithm(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.sca_custom.run()
		self.assertTrue(x)
		y = self.sca_customc.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.sca_griewank.run()
		self.assertTrue(x)
		y = self.sca_griewankc.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
