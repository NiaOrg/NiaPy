# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements
from unittest import TestCase
from numpy import random as rnd, full, inf
from NiaPy.benchmarks.utility import Task
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import TwoPointCrossover, MultiPointCrossover, CreepMutation, Chromosome

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

class ChromosomeTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), Task(self.D, 230, None, MyBenchmark())
		self.s1, self.s2, self.s3 = Chromosome(x=self.x), Chromosome(task=self.task, rand=rnd), Chromosome(task=self.task)

	def test_x_fine(self):
		self.assertTrue(False not in self.x == self.s1.x)

	def test_generateSolutin_fine(self):
		self.assertTrue(self.task.isFeasible(self.s2))
		self.assertTrue(self.task.isFeasible(self.s3))

	def test_evaluate_fine(self):
		self.s1.evaluate(self.task)
		self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

	def test_repair_fine(self):
		s = Chromosome(x=full(self.D, 100))
		self.assertFalse(self.task.isFeasible(s.x))
		s.repair(self.task)
		self.assertTrue(self.task.isFeasible(s.x))

	def test_eq_fine(self):
		self.assertFalse(self.s1 == self.s2)
		self.assertTrue(self.s1 == self.s1)
		s = Chromosome(x=self.s1.x)
		self.assertTrue(s == self.s1)

	def test_str_fine(self):
		self.assertEqual(str(self.s1), '%s -> %s' % (self.x, inf))

	def test_getitem_fine(self):
		for i in range(self.D): self.assertEqual(self.s1[i], self.x[i])

	def test_len_fine(self):
		self.assertEqual(len(self.s1), len(self.x))

class GATestCase(TestCase):
	def setUp(self):
		self.ga_custom = GeneticAlgorithm(D=10, NP=40, nFES=1000, Ts=4, Mr=0.05, Cr=0.4, benchmark=MyBenchmark())
		self.ga_griewank = GeneticAlgorithm(D=10, NP=40, nFES=1000, Ts=4, Mr=0.05, Cr=0.4, benchmark='griewank')
		self.ga_tpcr = GeneticAlgorithm(D=10, NP=40, nFES=1000, Ts=4, Mr=0.05, Cr=0.4, Crossover=TwoPointCrossover, benchmark='griewank')
		self.ga_mpcr = GeneticAlgorithm(D=10, NP=40, nFES=1000, Ts=4, Mr=0.05, Cr=4, Crossover=MultiPointCrossover, benchmark='griewank')
		self.ga_crmt = GeneticAlgorithm(D=10, NP=40, nFES=1000, Ts=4, Mr=0.05, Cr=0.4, Mutation=CreepMutation, benchmark='griewank')

	def test_custom_works_fine(self):
		self.assertTrue(self.ga_custom.run())

	def test_griewank_works_fine(self):
		self.assertTrue(self.ga_griewank.run())

	def test_two_point_crossover_fine(self):
		self.assertTrue(self.ga_tpcr.run())

	def test_multi_point_crossover_fine(self):
		self.assertTrue(self.ga_mpcr.run())

	def test_creep_mutation_fine(self):
		self.assertTrue(self.ga_crmt.run())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
