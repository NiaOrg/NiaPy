# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import NelderMeadMethod

class NMMTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		nmm_custom = NelderMeadMethod(n=10, C_a=2, C_r=0.5, seed=self.seed)
		nmm_customc = NelderMeadMethod(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, nmm_custom, nmm_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		nmm_griewank = NelderMeadMethod(n=10, C_a=5, C_r=0.5, seed=self.seed)
		nmm_griewankc = NelderMeadMethod(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, nmm_griewank, nmm_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
