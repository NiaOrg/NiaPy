# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.modified import DifferentialEvolutionMTS, DifferentialEvolutionMTSv1, MultiStratgyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTSv1, DynNpDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTSv1

class DEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DynNpDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DynNpDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DynNpDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpDEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DynNpDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DynNpDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DynNpDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class MSDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = MultiStratgyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = MultiStratgyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = MultiStratgyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = MultiStratgyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class MSDEMTSv1STestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpMSDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpMSDEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		ca_customc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=MyBenchmark(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc)

	def test_griewank_works_fine(self):
		ca_griewank = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		ca_griewankc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, D=self.D, nGEN=self.nGEN, nFES=self.nFES, benchmark=Griewank(), seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
