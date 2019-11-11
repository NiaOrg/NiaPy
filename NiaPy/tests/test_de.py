# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import DifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, MultiStrategyDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution, AgingNpMultiMutationDifferentialEvolution
from NiaPy.algorithms.basic.de import CrossRand1, CrossRand2, CrossBest1, CrossBest2, CrossCurr2Rand1, CrossCurr2Best1

class DETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc)

	def test_CrossRand1(self):
		de_rand1 = self.algo(CrossMutt=CrossRand1, seed=self.seed)
		de_rand1c = self.algo(CrossMutt=CrossRand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_rand1, de_rand1c)

	def test_CrossBest1(self):
		de_best1 = self.algo(CrossMutt=CrossBest1, seed=self.seed)
		de_best1c = self.algo(CrossMutt=CrossBest1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_best1, de_best1c)

	def test_CrossRand2(self):
		de_rand2 = self.algo(CrossMutt=CrossRand2, seed=self.seed)
		de_rand2c = self.algo(CrossMutt=CrossRand2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_rand2, de_rand2c)

	def test_CrossBest2(self):
		de_best2 = self.algo(CrossMutt=CrossBest2, seed=self.seed)
		de_best2c = self.algo(CrossMutt=CrossBest2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_best2, de_best2c)

	def test_CrossCurr2Rand1(self):
		de_curr2rand1 = self.algo(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		de_curr2rand1c = self.algo(CrossMutt=CrossCurr2Rand1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_curr2rand1, de_curr2rand1c)

	def test_CrossCurr2Best1(self):
		de_curr2best1 = self.algo(CrossMutt=CrossCurr2Best1, seed=self.seed)
		de_curr2best1c = self.algo(CrossMutt=CrossCurr2Best1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_curr2best1, de_curr2best1c)

class dynNpDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpDifferentialEvolution

	def test_typeParameters(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

class ANpDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AgingNpDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

class MsDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MultiStrategyDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = MultiStrategyDifferentialEvolution(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = MultiStrategyDifferentialEvolution(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

class dynNpMsDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynNpMultiStrategyDifferentialEvolution

	def test_typeParameters(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['rp'](10))
		self.assertTrue(d['rp'](10.10))
		self.assertFalse(d['rp'](0))
		self.assertFalse(d['rp'](-10))
		self.assertTrue(d['pmax'](10))
		self.assertFalse(d['pmax'](0))
		self.assertFalse(d['pmax'](-10))
		self.assertFalse(d['pmax'](10.12))

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

class ANpMsDETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AgingNpMultiMutationDifferentialEvolution

	def test_Custom_works_fine(self):
		de_custom = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		de_customc = self.algo(NP=40, F=0.5, CR=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		de_griewank = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		de_griewankc = self.algo(NP=10, CR=0.5, F=0.9, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
