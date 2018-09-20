# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import DifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1

class DETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = DifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		de_customc = DifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc)

	def test_griewank_works_fine(self):
		de_griewank = DifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		de_griewankc = DifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

	def test_CrossRand1(self):
		de_rand1 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossRand1, seed=self.seed)
		de_rand1c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossRand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand1, de_rand1c)

	def test_CrossBest1(self):
		de_best1 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossBest1, seed=self.seed)
		de_best1c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossBest1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best1, de_best1c)

	def test_CrossRand2(self):
		de_rand2 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossRand2, seed=self.seed)
		de_rand2c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossRand2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_rand2, de_rand2c)

	def test_CrossBest2(self):
		de_best2 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossBest2, seed=self.seed)
		de_best2c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossBest2, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_best2, de_best2c)

	def test_CrossCurr2Rand1(self):
		de_curr2rand1 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossCurr2Rand1, seed=self.seed)
		de_curr2rand1c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossCurr2Rand1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_curr2rand1, de_curr2rand1c)

	def test_CrossCurr2Best1(self):
		de_curr2best1 = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossCurr2Best1, seed=self.seed)
		de_curr2best1c = DifferentialEvolution(nFES=self.nFES, nGEN=self.nGEN, D=self.D, CrossMutt=CrossCurr2Best1, seed=self.seed)
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
		de_custom = DynNpDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		de_customc = DynNpDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc)

	def test_griewank_works_fine(self):
		de_griewank = DynNpDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		de_griewankc = DynNpDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

class ANpDETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = AgingNpDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		de_customc = AgingNpDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc)

	def test_griewank_works_fine(self):
		de_griewank = AgingNpDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		de_griewankc = AgingNpDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

class MsDETestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		de_custom = MultiStrategyDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		de_customc = MultiStrategyDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc)

	def test_griewank_works_fine(self):
		de_griewank = MultiStrategyDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		de_griewankc = MultiStrategyDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

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
		de_custom = DynNpMultiStrategyDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		de_customc = DynNpMultiStrategyDifferentialEvolution(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, F=0.5, CR=0.9, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_custom, de_customc)

	def test_griewank_works_fine(self):
		de_griewank = DynNpMultiStrategyDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		de_griewankc = DynNpMultiStrategyDifferentialEvolution(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, CR=0.5, F=0.9, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, de_griewank, de_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
