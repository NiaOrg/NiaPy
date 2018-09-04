# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import CamelAlgorithm

class CATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = CamelAlgorithm(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = CamelAlgorithm(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = CamelAlgorithm(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = CamelAlgorithm(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
