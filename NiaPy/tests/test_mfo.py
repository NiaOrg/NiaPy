# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import MothFlameOptimizer

class MFOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = MothFlameOptimizer

	def test_custom_works_fine(self):
		mfo_custom = self.algo(NP=20, seed=self.seed)
		mfo_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mfo_custom, mfo_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mfo_griewank = self.algo(NP=10, seed=self.seed)
		mfo_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, mfo_griewank, mfo_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
