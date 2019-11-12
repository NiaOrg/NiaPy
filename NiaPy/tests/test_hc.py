# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.other import HillClimbAlgorithm

class HCTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HillClimbAlgorithm

	def test_custom_works_fine(self):
		ihc_custom = self.algo(delta=0.4, seed=self.seed)
		ihc_customc = self.algo(delta=0.4, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ihc_custom, ihc_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ihc_griewank = self.algo(delta=0.1, seed=self.seed)
		ihc_griewankc = self.algo(delta=0.1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ihc_griewank, ihc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
