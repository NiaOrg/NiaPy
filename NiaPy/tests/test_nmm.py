# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import NelderMeadMethod

class NMMTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		nmm_custom = NelderMeadMethod(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		nmm_customc = NelderMeadMethod(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, nmm_custom, nmm_customc)

	def test_griewank_works_fine(self):
		nmm_griewank = NelderMeadMethod(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		nmm_griewankc = NelderMeadMethod(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, nmm_griewank, nmm_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
