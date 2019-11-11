# encoding=utf8
from NiaPy.algorithms.basic import GravitationalSearchAlgorithm
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GSATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GravitationalSearchAlgorithm

	def test_Custom_works_fine(self):
		gsa_custom = self.algo(NP=40, seed=self.seed)
		gsa_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gsa_custom, gsa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gsa_griewank = self.algo(NP=10, seed=self.seed)
		gsa_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gsa_griewank, gsa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
