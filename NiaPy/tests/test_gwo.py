# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import GreyWolfOptimizer


class GWOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GreyWolfOptimizer

	def test_custom_works_fine(self):
		gwo_custom = self.algo(NP=20, seed=self.seed)
		gwo_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gwo_custom, gwo_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gwo_griewank = self.algo(NP=10, seed=self.seed)
		gwo_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gwo_griewank, gwo_griewankc)
