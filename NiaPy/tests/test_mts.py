# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from numpy import array_equal
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1

class MTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mts_custom = MultipleTrajectorySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		mts_customc = MultipleTrajectorySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc)

	def test_griewank_works_fine(self):
		mts_griewank = MultipleTrajectorySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		mts_griewankc = MultipleTrajectorySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

class MTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mts_custom = MultipleTrajectorySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		mts_customc = MultipleTrajectorySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc)

	def test_griewank_works_fine(self):
		mts_griewank = MultipleTrajectorySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		mts_griewankc = MultipleTrajectorySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
