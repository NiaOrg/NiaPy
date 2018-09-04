# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class GSOTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimization(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		gso_customc = GlowwormSwarmOptimization(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc)

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimization(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimization(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc)

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV1(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv2TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV2(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV2(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc)

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV2(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV2(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

class GSOv3TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		gso_custom = GlowwormSwarmOptimizationV3(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		gso_customc = GlowwormSwarmOptimizationV3(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_custom, gso_customc)

	def test_griewank_works_fine(self):
		gso_griewank = GlowwormSwarmOptimizationV3(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		gso_griewankc = GlowwormSwarmOptimizationV3(D=self.D, nFES=self.nFES, nGEN=self.nGEN, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, gso_griewank, gso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
