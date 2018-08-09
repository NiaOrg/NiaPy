# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements, old-style-class
from unittest import TestCase
from numpy import random as rnd, full, inf, array_equal
from NiaPy.benchmarks.utility import Task
from NiaPy.algorithms.algorithm import Algorithm, Individual

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

class IndividualTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), Task(self.D, 230, None, MyBenchmark())
		self.s1, self.s2, self.s3 = Individual(x=self.x), Individual(task=self.task, rand=rnd), Individual(task=self.task)

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

class AlgorithmTestCase(TestCase):
	def setUp(self):
		self.a = Algorithm()

	def test__fine(self):
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
