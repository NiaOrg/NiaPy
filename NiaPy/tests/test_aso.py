# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.algorithms.other.aso import Elitism, Sequential, Crossover

class ASOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AnarchicSocietyOptimization

	def test_parameter_types(self):
		d = self.algo.typeParameters()
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

class ASOElitismTestCase(ASOTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=Elitism, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = self.algo(NP=40, Combination=Elitism, seed=self.seed)
		aso_griewankc = self.algo(NP=40, Combination=Elitism, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)

class ASOSequentialTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=Sequential, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = self.algo(NP=40, Combination=Sequential, seed=self.seed)
		aso_griewankc = self.algo(NP=40, Combination=Sequential, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)

class ASOCrossoverTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		aso_custom = self.algo(NP=40, Combination=Crossover, seed=self.seed)
		aso_customc = self.algo(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aso_griewank = self.algo(NP=40, Combination=Crossover, seed=self.seed)
		aso_griewankc = self.algo(NP=40, Combination=Crossover, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
