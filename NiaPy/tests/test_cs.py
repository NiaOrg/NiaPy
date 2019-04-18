# encoding=utf8
# pylint: disable=mixed-indentation, too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CuckooSearch

class CSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		cs_custom = CuckooSearch(N=20, seed=self.seed)
		cs_customc = CuckooSearch(N=20, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_custom, cs_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		cs_griewank = CuckooSearch(N=10, seed=self.seed)
		cs_griewankc = CuckooSearch(N=10, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_griewank, cs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
