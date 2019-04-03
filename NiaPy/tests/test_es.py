# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from unittest import TestCase
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML, CovarianceMatrixAdaptionEvolutionStrategy
from NiaPy.algorithms.basic.es import IndividualES

class IndividualESTestCase(TestCase):
	def test_init_ok_one(self):
		i = IndividualES()
		self.assertEqual(i.rho, 1.0)

	def test_init_ok_two(self):
		i = IndividualES(rho=10)
		self.assertEqual(i.rho, 10)

class ES1p1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		es_custom = EvolutionStrategy1p1(k=10, c_a=1.5, c_r=0.42, seed=self.seed)
		es_customc = EvolutionStrategy1p1(k=10, c_a=1.5, c_r=0.42, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategy1p1(k=15, c_a=1.2, c_r=0.5, seed=self.seed)
		es_griewankc = EvolutionStrategy1p1(k=15, c_a=1.2, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

class ESMp1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyMp1(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = EvolutionStrategyMp1(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyMp1(mu=30, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		es_griewankc = EvolutionStrategyMp1(mu=30, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

class ESMpLTestCase(AlgorithmTestCase):
	def test_typeParametes(self):
		d = EvolutionStrategyML.typeParameters()
		self.assertTrue(d['lam'](10))
		self.assertFalse(d['lam'](10.10))
		self.assertFalse(d['lam'](0))
		self.assertFalse(d['lam'](-10))

	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyMpL(mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = EvolutionStrategyMpL(mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyMpL(mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		es_griewankc = EvolutionStrategyMpL(mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = EvolutionStrategyMpL(mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es1_customc = EvolutionStrategyMpL(mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc, MyBenchmark())

	def test_griewank1_works_fine(self):
		es1_griewank = EvolutionStrategyMpL(mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		es1_griewankc = EvolutionStrategyMpL(mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

class ESMLTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyML(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es_customc = EvolutionStrategyML(mu=45, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyML(mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, seed=self.seed)
		es_griewankc = EvolutionStrategyML(mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = EvolutionStrategyML(mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		es1_customc = EvolutionStrategyML(mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc, MyBenchmark())

	def test_griewank1_works_fine(self):
		es1_griewank = EvolutionStrategyML(mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		es1_griewankc = EvolutionStrategyML(mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

class CMAESTestCase(AlgorithmTestCase):
	def test_typeParametes(self):
		d = CovarianceMatrixAdaptionEvolutionStrategy.typeParameters()
		self.assertTrue(d['epsilon'](0.234))
		self.assertFalse(d['epsilon'](-0.234))
		self.assertFalse(d['epsilon'](10000.234))
		self.assertFalse(d['epsilon'](10))

	def test_custom_works_fine(self):
		es_custom = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		es_customc = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		es_griewank = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		es_griewankc = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		es1_customc = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc, MyBenchmark())

	def test_griewank1_works_fine(self):
		es1_griewank = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		es1_griewankc = CovarianceMatrixAdaptionEvolutionStrategy(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
