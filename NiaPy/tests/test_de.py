# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import DifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution, AgingNpMultiMutationDifferentialEvolution
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1

class DETestCase(AlgorithmTestCase):
	r"""Test case for DifferentialEvolution algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		:class:`NiaPy.algorithms.DifferentialEvolution`
	"""
	def test_manual_population_initialization(self):
		pass

	def test_Custom_works_fine(self):
		de_custom = DifferentialEvolution(F=0.5, CR=0.9, seed=self.seed)
		de_customc = DifferentialEvolution(F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = DifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

	def test_CrossRand1(self):
		de_rand1 = DifferentialEvolution(CrossMutt=CrossRand1, seed=self.seed)
		de_rand1c = DifferentialEvolution(CrossMutt=CrossRand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand1, de_rand1c)

	def test_CrossBest1(self):
		de_best1 = DifferentialEvolution(CrossMutt=CrossBest1, seed=self.seed)
		de_best1c = DifferentialEvolution(CrossMutt=CrossBest1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best1, de_best1c)

	def test_CrossRand2(self):
		de_rand2 = DifferentialEvolution(CrossMutt=CrossRand2, seed=self.seed)
		de_rand2c = DifferentialEvolution(CrossMutt=CrossRand2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand2, de_rand2c)

	def test_CrossBest2(self):
		de_best2 = DifferentialEvolution(CrossMutt=CrossBest2, seed=self.seed)
		de_best2c = DifferentialEvolution(CrossMutt=CrossBest2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best2, de_best2c)

	def test_CrossCurr2Rand1(self):
		de_curr2rand1 = DifferentialEvolution(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		de_curr2rand1c = DifferentialEvolution(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_curr2rand1, de_curr2rand1c)

	def test_CrossCurr2Best1(self):
		de_curr2best1 = DifferentialEvolution(CrossMutt=CrossCurr2Best1, seed=self.seed)
		de_curr2best1c = DifferentialEvolution(CrossMutt=CrossCurr2Best1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_curr2best1, de_curr2best1c)

class dynNpDETestCase(AlgorithmTestCase):
	def test_typeParameters(self):
		d = DynNpDifferentialEvolution.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = DynNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = DynNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = DynNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DynNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class ANpDETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = AgingNpDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = AgingNpDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class MsDETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class dynNpMsDETestCase(AlgorithmTestCase):
	def test_typeParameters(self):
		d = DynNpMultiStrategyDifferentialEvolution.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = DynNpMultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = DynNpMultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = DynNpMultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = DynNpMultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

class ANpMsDETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = AgingNpMultiMutationDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = AgingNpMultiMutationDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = AgingNpMultiMutationDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = AgingNpMultiMutationDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc, 'griewank')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
