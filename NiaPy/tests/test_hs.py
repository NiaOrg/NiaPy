# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import HarmonySearch, HarmonySearchV1

class HSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		hs_costom = HarmonySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=MyBenchmark())
		hs_costomc = HarmonySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=MyBenchmark())
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc)

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=Griewank())
		hs_griewankc = HarmonySearch(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=Griewank())
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

class HSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		hs_costom = HarmonySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=MyBenchmark())
		hs_costomc = HarmonySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=MyBenchmark())
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc)

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=Griewank())
		hs_griewankc = HarmonySearchV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, seed=self.seed, benchmark=Griewank())
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
