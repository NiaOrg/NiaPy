# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import HarmonySearch, HarmonySearchV1

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

class HSTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.hs_custom = HarmonySearch(D=self.D, nFES=1000, benchmark=MyBenchmark())
		self.hs_griewank = HarmonySearch(D=self.D, nFES=1000, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.hs_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.hs_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class HSV1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.hs_custom = HarmonySearchV1(D=self.D, nFES=1000, benchmark=MyBenchmark())
		self.hs_griewank = HarmonySearchV1(D=self.D, nFES=1000, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.hs_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.hs_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
