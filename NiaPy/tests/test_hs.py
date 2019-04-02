# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import HarmonySearch, HarmonySearchV1

class HSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		hs_costom = HarmonySearch(seed=self.seed)
		hs_costomc = HarmonySearch(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc, MyBenchmark())

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearch(seed=self.seed)
		hs_griewankc = HarmonySearch(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

class HSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		hs_costom = HarmonySearchV1(seed=self.seed)
		hs_costomc = HarmonySearchV1(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc, MyBenchmark())

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearchV1(seed=self.seed)
		hs_griewankc = HarmonySearchV1(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
