# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, too-many-function-args
from unittest import TestCase

from numpy import array_equal, full, inf, random as rnd

from NiaPy.task import Task
from NiaPy.algorithms.basic import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.mke import MkeSolution
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class MkeSolutionTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-2, 2, self.D), Task(self.D, nGEN=230, nFES=inf, benchmark=MyBenchmark())
		self.sol1, self.sol2, self.sol3 = MkeSolution(x=self.x, e=False), MkeSolution(task=self.task), MkeSolution(x=self.x, e=False)

	def test_uPersonalBest_fine(self):
		self.sol2.uPersonalBest()
		self.assertTrue(array_equal(self.sol2.x, self.sol2.x_pb))
		self.assertEqual(self.sol2.f_pb, self.sol2.f)
		self.sol3.evaluate(self.task)
		self.sol3.x = full(self.task.D, -5.11)
		self.sol3.evaluate(self.task)
		self.sol3.uPersonalBest()
		self.assertTrue(array_equal(self.sol3.x, self.sol3.x_pb))
		self.assertEqual(self.sol3.f_pb, self.sol3.f)

class MKEv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mke_custom = MonkeyKingEvolutionV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = MonkeyKingEvolutionV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_custom, mke_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mke_griewank = MonkeyKingEvolutionV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mke_griewankc = MonkeyKingEvolutionV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_griewank, mke_griewankc)

class MKEv2TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mke_custom = MonkeyKingEvolutionV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = MonkeyKingEvolutionV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_custom, mke_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mke_griewank = MonkeyKingEvolutionV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mke_griewankc = MonkeyKingEvolutionV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_griewank, mke_griewankc)

class MKEv3TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mke_custom = MonkeyKingEvolutionV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mke_customc = MonkeyKingEvolutionV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_custom, mke_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mke_griewank = MonkeyKingEvolutionV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mke_griewankc = MonkeyKingEvolutionV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mke_griewank, mke_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
