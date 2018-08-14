# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1

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

class MTSTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.mts_custom = MultipleTrajectorySearch(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.mts_griewank = MultipleTrajectorySearch(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.mts_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.mts_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class MTSv1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.mts_custom = MultipleTrajectorySearchV1(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.mts_griewank = MultipleTrajectorySearchV1(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.mts_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.mts_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
