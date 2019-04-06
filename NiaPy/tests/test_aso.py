# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from unittest import TestCase

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.algorithms.other.aso import Elitism, Sequential, Crossover

class ASOTestCase(TestCase):
	def test_parameter_types(self):
		d = AnarchicSocietyOptimization.typeParameters()
		self.assertTrue(d['NP'](1))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-1))
		self.assertTrue(d['F'](10))
		self.assertFalse(d['F'](0))
		self.assertFalse(d['F'](-10))
		self.assertTrue(d['CR'](0.1))
		self.assertFalse(d['CR'](-19))
		self.assertFalse(d['CR'](19))
		self.assertTrue(d['alpha'](10))
		self.assertTrue(d['gamma'](10))
		self.assertTrue(d['theta'](10))

class ASOElitismTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

class ASOSequentialTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

class ASOCrossoverTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		aso_customc = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		aso_griewankc = AnarchicSocietyOptimization(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, aso_griewank, aso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
