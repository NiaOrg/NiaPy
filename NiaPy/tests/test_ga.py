# encoding=utf8
# pylint: disable=mixed-indentation, function-redefined, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.algorithms.basic import GeneticAlgorithm
from NiaPy.algorithms.basic.ga import TwoPointCrossover, MultiPointCrossover, CreepMutation

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
