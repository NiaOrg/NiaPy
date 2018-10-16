# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.algorithms.basic import GravitationalSearchAlgorithm
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GSATestCase(AlgorithmTestCase):
	def test_Custom_works_fine(self):
		gsa_custom = GravitationalSearchAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		gsa_customc = GravitationalSearchAlgorithm(D=self.D, NP=40, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gsa_custom, gsa_customc)

	def test_griewank_works_fine(self):
		gsa_griewank = GravitationalSearchAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		gsa_griewankc = GravitationalSearchAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gsa_griewank, gsa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
