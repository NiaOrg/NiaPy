# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.other import HillClimbAlgorithm

class HCTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ihc_custom = HillClimbAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, delta=0.4, benchmark=MyBenchmark(), seed=self.seed)
		ihc_customc = HillClimbAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, delta=0.4, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_custom, ihc_customc)

	def test_griewank_works_fine(self):
		ihc_griewank = HillClimbAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		ihc_griewankc = HillClimbAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_griewank, ihc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
