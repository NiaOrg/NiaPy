# encoding=utf8
# pylint: disable=mixed-indentation, too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CuckooSearch

class CSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		cs_custom = CuckooSearch(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		cs_customc = CuckooSearch(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_custom, cs_customc)

	def test_griewank_works_fine(self):
		cs_griewank = CuckooSearch(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		cs_griewankc = CuckooSearch(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cs_griewank, cs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
