# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm, FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss

class BBFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BareBonesFireworksAlgorithm

	def test_custom_works_fine(self):
		bbfwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		bbfwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bbfwa_custom, bbfwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		bbfwa_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		bbfwa_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bbfwa_griewank, bbfwa_griewankc)

class FWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)

class EFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = EnhancedFireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)

class DFWATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynamicFireworksAlgorithm

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)

class DFWAGTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = DynamicFireworksAlgorithmGauss

	def test_custom_works_fine(self):
		fwa_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		fwa_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_custom, fwa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fwa_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		fwa_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fwa_griewank, fwa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
