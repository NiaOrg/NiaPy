# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from numpy import asarray
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.algorithms.other.aso import Elitism, Sequential, Crossover

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

class ASOElitismTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.aso_custom = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Elitism, benchmark=MyBenchmark())
		self.aso_griewank = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Elitism, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.aso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.aso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)

class ASOSequentialTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.aso_custom = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Sequential, benchmark=MyBenchmark())
		self.aso_griewank = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Sequential, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.aso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.aso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)

class ASOCrossoverTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.aso_custom = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Crossover, benchmark=MyBenchmark())
		self.aso_griewank = AnarchicSocietyOptimization(NP=40, D=self.D, nGEN=10, nFES=4000, Combination=Crossover, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.aso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.aso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, asarray(x[0])), x[1], delta=1e3)


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
