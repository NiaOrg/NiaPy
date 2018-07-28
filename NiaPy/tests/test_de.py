# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from unittest import TestCase
from numpy import random as rnd, full, asarray, inf
from NiaPy.benchmarks.utility import Task
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1, SolutionDE

class MyBenchmark(object):
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

class SolutionDETestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), Task(self.D, 230, None, MyBenchmark())
		self.s1, self.s2, self.s3 = SolutionDE(x=self.x), SolutionDE(task=self.task, rand=rnd), SolutionDE(task=self.task)

	def test_x_fine(self):
		self.assertTrue(False not in self.x == self.s1.x)

	def test_generateSolutin_fine(self):
		self.assertTrue(self.task.isFeasible(self.s2))
		self.assertTrue(self.task.isFeasible(self.s3))

	def test_evaluate_fine(self):
		self.s1.evaluate(self.task)
		self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

	def test_repair_fine(self):
		s = SolutionDE(x=full(self.D, 100))
		self.assertFalse(self.task.isFeasible(s.x))
		s.repair(self.task)
		self.assertTrue(self.task.isFeasible(s.x))

	def test_eq_fine(self):
		self.assertFalse(self.s1 == self.s2)
		self.assertTrue(self.s1 == self.s1)
		s = SolutionDE(x=self.s1.x)
		self.assertTrue(s == self.s1)

	def test_str_fine(self):
		self.assertEqual(str(self.s1), '%s -> %s' % (self.x, inf))

	def test_getitem_fine(self):
		for i in range(self.D): self.assertEqual(self.s1[i], self.x[i])

	def test_len_fine(self):
		self.assertEqual(len(self.s1), len(self.x))


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
