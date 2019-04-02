# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11

class KHV1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		kh_custom = KrillHerdV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV2TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		kh_custom = KrillHerdV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV2(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV2(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV3TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		kh_custom = KrillHerdV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV4TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		kh_custom = KrillHerdV4(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV4(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV4(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV4(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV11TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		kh_custom = KrillHerdV11(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV11(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV11(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV11(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
