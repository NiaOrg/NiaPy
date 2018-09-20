# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks import Griewank
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm, FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss

class BBFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		bbfwa_custom = BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		bbfwa_customc = BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bbfwa_custom, bbfwa_customc)

	def test_griewank_works_fine(self):
		bbfwa_griewank = BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		bbfwa_griewankc = BareBonesFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, bbfwa_griewank, bbfwa_griewankc)

class FWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = FireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		fwa_customc = FireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc)

	def test_griewank_works_fine(self):
		fwa_griewank = FireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		fwa_griewankc = FireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class EFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = EnhancedFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		fwa_customc = EnhancedFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc)

	def test_griewank_works_fine(self):
		fwa_griewank = EnhancedFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		fwa_griewankc = EnhancedFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class DFWATestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = DynamicFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		fwa_customc = DynamicFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc)

	def test_griewank_works_fine(self):
		fwa_griewank = DynamicFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		fwa_griewankc = DynamicFireworksAlgorithm(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

class DFWAGTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		fwa_custom = DynamicFireworksAlgorithmGauss(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		fwa_customc = DynamicFireworksAlgorithmGauss(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_custom, fwa_customc)

	def test_griewank_works_fine(self):
		fwa_griewank = DynamicFireworksAlgorithmGauss(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		fwa_griewankc = DynamicFireworksAlgorithmGauss(D=self.D, nFES=self.nFES, nGEN=self.nGEN, n=10, C_a=5, C_r=0.5, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, fwa_griewank, fwa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
