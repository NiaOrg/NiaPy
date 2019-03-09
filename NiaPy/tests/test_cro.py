# encoding=utf8
# pylint: disable=mixed-indentation, too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CoralReefsOptimization

# FIXME algorithm has some problems

class CROTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		cro_custom = CoralReefsOptimization(D=self.D, N=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		cro_customc = CoralReefsOptimization(D=self.D, N=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cro_custom, cro_customc)

	def test_griewank_works_fine(self):
		cro_griewank = CoralReefsOptimization(N=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		cro_griewankc = CoralReefsOptimization(N=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, cro_griewank, cro_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
