# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CuckooSearch

class CSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = CuckooSearch

	def test_custom_works_fine(self):
		cs_custom = self.algo(NP=20, seed=self.seed)
		cs_customc = self.algo(NP=20, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cs_custom, cs_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		cs_griewank = self.algo(NP=10, seed=self.seed)
		cs_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, cs_griewank, cs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
