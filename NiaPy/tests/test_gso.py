# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
from NiaPy.algorithms.basic import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GSOTestCase(AlgorithmTestCase):
	def test_algorithm_info_fine(self):
		self.assertIsNotNone(GlowwormSwarmOptimization.algorithmInfo())

	def test_type_parameters_fine(self):
		d = GlowwormSwarmOptimization.typeParameters()
		self.assertIsNotNone(d.get('n', None))
		self.assertIsNotNone(d.get('l0', None))
		self.assertIsNotNone(d.get('nt', None))
		self.assertIsNotNone(d.get('rho', None))
		self.assertIsNotNone(d.get('gamma', None))
		self.assertIsNotNone(d.get('beta', None))
		self.assertIsNotNone(d.get('s', None))

	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimization(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = GlowwormSwarmOptimization(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimization(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimization(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV1(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV1(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV1(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV1(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv2TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV2(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV2(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV2(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV2(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv3TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV3(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV3(n=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV3(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV3(n=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
