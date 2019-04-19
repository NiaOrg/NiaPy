# encoding=utf8

from numpy import random as rnd, array_equal

from NiaPy.algorithms.modified import DifferentialEvolutionMTS, DifferentialEvolutionMTSv1, MultiStrategyDifferentialEvolutionMTS, MultiStrategyDifferentialEvolutionMTSv1, DynNpMultiStrategyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTSv1, DynNpDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTSv1
from NiaPy.algorithms.modified.hde import MtsIndividual

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark, IndividualTestCase

class MtsIndividualTestCase(IndividualTestCase):
	def setUp(self):
		IndividualTestCase.setUp(self)
		self.s1, self.s2, self.s3, self.s4 = MtsIndividual(x=self.x, task=self.task, e=False), MtsIndividual(task=self.task, SR=self.task.bRange / 10, rand=rnd), MtsIndividual(task=self.task), MtsIndividual()

	def test_default_values_init_ok(self):
		self.assertIsNone(self.s4.SR)
		self.assertTrue(array_equal(self.task.bRange / 10, self.s2.SR))
		self.assertTrue(array_equal(self.task.bRange / 4, self.s1.SR))
		self.assertEqual(0, self.s1.grade)
		self.assertTrue(self.s1.enable)
		self.assertFalse(self.s1.improved)

class DEMTSTestCase(AlgorithmTestCase):
	def test_type_parameters_fine(self):
		d = DifferentialEvolutionMTS.typeParameters()
		self.assertIsNotNone(d.get('NoEnabled', None))
		self.assertIsNotNone(d.get('NoLs', None))
		self.assertIsNotNone(d.get('NoLsTests', None))

	def test_custom_works_fine(self):
		ca_custom = DifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_customc = DifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_griewankc = DifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_customc = DifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_griewankc = DifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_customc = DynNpDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DynNpDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_griewankc = DynNpDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpDEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_customc = DynNpDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DynNpDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_griewankc = DynNpDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class MSDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = MultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_customc = MultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = MultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_griewankc = MultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class MSDEMTSv1STestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = MultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_customc = MultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = MultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_griewankc = MultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpMSDEMTSTestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_customc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		ca_griewankc = DynNpMultiStrategyDifferentialEvolutionMTS(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

class DynNpMSDEMTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		ca_custom = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_customc = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		ca_griewankc = DynNpMultiStrategyDifferentialEvolutionMTSv1(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
