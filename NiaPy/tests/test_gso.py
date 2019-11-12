# encoding=utf8

from NiaPy.algorithms.basic import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GSOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimization

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_griewank, gso_griewankc)

class GSOv1TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV1

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_griewank, gso_griewankc)

class GSOv2TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV2

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_griewank, gso_griewankc)

class GSOv3TestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = GlowwormSwarmOptimizationV3

	def test_custom_works_fine(self):
		gso_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		gso_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_custom, gso_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		gso_griewank = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		gso_griewankc = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, gso_griewank, gso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
