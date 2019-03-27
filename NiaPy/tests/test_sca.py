# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import SineCosineAlgorithm
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class SCATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		sca_custom = SineCosineAlgorithm(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		sca_customc = SineCosineAlgorithm(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, sca_custom, sca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		sca_griewank = SineCosineAlgorithm(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		sca_griewankc = SineCosineAlgorithm(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, sca_griewank, sca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
