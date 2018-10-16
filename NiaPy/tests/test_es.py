# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from unittest import TestCase
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML, CovarianceMaatrixAdaptionEvolutionStrategy
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
		es_custom = EvolutionStrategy1p1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, k=10, c_a=1.5, c_r=0.42, benchmark=MyBenchmark(), seed=self.seed)
		es_customc = EvolutionStrategy1p1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, k=10, c_a=1.5, c_r=0.42, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc)

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategy1p1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, k=15, c_a=1.2, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es_griewankc = EvolutionStrategy1p1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, k=15, c_a=1.2, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

class ESMp1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyMp1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		es_customc = EvolutionStrategyMp1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc)

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyMp1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es_griewankc = EvolutionStrategyMp1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

class ESMpLTestCase(AlgorithmTestCase):
	def test_typeParametes(self):
		d = EvolutionStrategyML.typeParameters()
		self.assertTrue(d['lam'](10))
		self.assertFalse(d['lam'](10.10))
		self.assertFalse(d['lam'](0))
		self.assertFalse(d['lam'](-10))

	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		es_customc = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc)

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es_griewankc = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		es1_customc = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc)

	def test_griewank1_works_fine(self):
		es1_griewank = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es1_griewankc = EvolutionStrategyMpL(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

class ESMLTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		es_custom = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		es_customc = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc)

	def test_griewank_works_fine(self):
		es_griewank = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es_griewankc = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		es1_customc = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc)

	def test_griewank1_works_fine(self):
		es1_griewank = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		es1_griewankc = EvolutionStrategyML(D=self.D, nFES=self.nFES, nGEN=self.nGEN, mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

class CMAESTestCase(AlgorithmTestCase):
	def test_typeParametes(self):
		d = CovarianceMaatrixAdaptionEvolutionStrategy.typeParameters()
		self.assertTrue(d['epsilon'](0.234))
		self.assertFalse(d['epsilon'](-0.234))
		self.assertFalse(d['epsilon'](10000.234))
		self.assertFalse(d['epsilon'](10))

	def test_custom_works_fine(self):
		es_custom = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		es_customc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_custom, es_customc)

	def test_griewank_works_fine(self):
		es_griewank = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		es_griewankc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es_griewank, es_griewankc)

	def test_custom1_works_fine(self):
		es1_custom = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		es1_customc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_custom, es1_customc)

	def test_griewank1_works_fine(self):
		es1_griewank = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		es1_griewankc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, es1_griewank, es1_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
