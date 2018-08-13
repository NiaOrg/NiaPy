# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from numpy import asarray
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1

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

class DETestCase(TestCase):
	def setUp(self):
		self.de_custom = (10, DifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, CR=0.9, benchmark=MyBenchmark()))
		self.de_griewank = (40, DifferentialEvolutionAlgorithm(NP=10, D=40, nFES=1000, CR=0.5, F=0.9, benchmark='griewank'))
		self.de_rand1 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossRand1))
		self.de_best1 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossBest1))
		self.de_rand2 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossRand2))
		self.de_best2 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossBest2))
		self.de_curr2rand1 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossCurr2Rand1))
		self.de_curr2best1 = (10, DifferentialEvolutionAlgorithm(nFES=1000, D=10, CrossMutt=CrossCurr2Best1))

	def test_Custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.de_custom[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_custom[0], asarray(x[0])), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.de_griewank[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_griewank[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossRand1(self):
		fun = Ackley().function()
		x = self.de_rand1[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_rand1[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossBest1(self):
		fun = Ackley().function()
		x = self.de_best1[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_best1[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossRand2(self):
		fun = Ackley().function()
		x = self.de_rand2[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_best2[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossBest2(self):
		fun = Ackley().function()
		x = self.de_best2[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_best2[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossCurr2Rand1(self):
		fun = Ackley().function()
		x = self.de_curr2rand1[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_curr2rand1[0], asarray(x[0])), x[1], delta=1e2)

	def test_CrossCurr2Best1(self):
		fun = Ackley().function()
		x = self.de_curr2best1[1].run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.de_curr2best1[0], asarray(x[0])), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
