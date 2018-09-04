# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm

class ABCTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		abc_custom = ArtificialBeeColonyAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		abc_customc = ArtificialBeeColonyAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, abc_custom, abc_customc)

	def test_griewank_works_fine(self):
		abc_griewank = ArtificialBeeColonyAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		abc_griewankc = ArtificialBeeColonyAlgorithm(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, abc_griewank, abc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
