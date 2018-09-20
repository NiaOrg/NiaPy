# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements, old-style-class, function-redefined
from unittest import TestCase
from numpy import random as rnd, full, inf, array_equal
from NiaPy.util import Task, OptimizationType
from NiaPy.algorithms.algorithm import Individual, Algorithm

class MyBenchmark:
	def __init__(self):
		self.Lower = -5.12
		self.Upper = 5.12
		self.optType = OptimizationType.MINIMIZATION

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class IndividualTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), Task(self.D, 230, inf, MyBenchmark())
		self.s1, self.s2, self.s3 = Individual(x=self.x, e=False), Individual(task=self.task, rand=rnd), Individual(task=self.task)

	def test_x_fine(self):
		self.assertTrue(array_equal(self.x, self.s1.x))

	def test_generateSolutin_fine(self):
		self.assertTrue(self.task.isFeasible(self.s2))
		self.assertTrue(self.task.isFeasible(self.s3))

	def test_evaluate_fine(self):
		self.s1.evaluate(self.task)
		self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

	def test_repair_fine(self):
		s = Individual(x=full(self.D, 100))
		self.assertFalse(self.task.isFeasible(s.x))
		s.repair(self.task)
		self.assertTrue(self.task.isFeasible(s.x))

	def test_eq_fine(self):
		self.assertFalse(self.s1 == self.s2)
		self.assertTrue(self.s1 == self.s1)
		s = Individual(x=self.s1.x)
		self.assertTrue(s == self.s1)

	def test_str_fine(self):
		self.assertEqual(str(self.s1), '%s -> %s' % (self.x, inf))

	def test_getitem_fine(self):
		for i in range(self.D): self.assertEqual(self.s1[i], self.x[i])

	def test_len_fine(self):
		self.assertEqual(len(self.s1), len(self.x))

class AlgorithBaseTestCase(TestCase):
	def setUp(self):
		self.algo = Algorithm()

	def test_setParameters(self):
		a = self.algo.setParameters(t=None, a=20)
		self.assertEqual(a, None)

	def test_setBenchmark(self):
		task = Task(D=10, nFES=10, nGEN=10, optType=OptimizationType.MINIMIZATION, benchmark=MyBenchmark())
		a = self.algo.setBechmark(task)
		self.assertIsInstance(a, Algorithm)

	def test_randn(self):
		a = self.algo.randn([1, 2])
		self.assertEqual(a.shape, (1, 2))
		a = self.algo.randn(1)
		self.assertEqual(len(a), 1)
		a = self.algo.randn(2)
		self.assertEqual(len(a), 2)
		a = self.algo.randn()
		self.assertIsInstance(a, float)

	def test_runYield(self):
		a = self.algo.runYield(None)
		self.assertEqual(next(a), (None, None))

	def test_runTask(self):
		a = self.algo.runTask(None)
		self.assertEqual(a, (None, None))

class AlgorithmTestCase(TestCase):
	def setUp(self):
		self.D, self.nGEN, self.nFES, self.seed = 40, 1000, 1000, 1

	def algorithm_run_test(self, a, b):
		x = a.run()
		self.assertTrue(x)
		y = b.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]), 'Results can not be reproduced, check usages of random number generator')
		self.assertEqual(x[1], y[1], 'Results can not be reproduced or bad function value')
		self.assertEqual(a.task.Iters, b.task.Iters)
		self.assertEqual(a.task.Evals, b.task.Evals)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
