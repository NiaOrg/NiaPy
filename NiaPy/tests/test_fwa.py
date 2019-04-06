# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm, FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss

class BBFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		bbfwa_custom = BareBonesFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		bbfwa_customc = BareBonesFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bbfwa_custom, bbfwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		bbfwa_griewank = BareBonesFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		bbfwa_griewankc = BareBonesFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bbfwa_griewank, bbfwa_griewankc)

class FWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = FireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = FireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = FireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = FireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class EFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = EnhancedFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = EnhancedFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = EnhancedFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = EnhancedFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class DFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = DynamicFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = DynamicFireworksAlgorithm(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = DynamicFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = DynamicFireworksAlgorithm(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class DFWAGTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = DynamicFireworksAlgorithmGauss(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = DynamicFireworksAlgorithmGauss(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = DynamicFireworksAlgorithmGauss(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = DynamicFireworksAlgorithmGauss(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
