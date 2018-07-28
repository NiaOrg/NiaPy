# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from unittest import TestCase
from numpy import random as rnd, inf, asarray
from NiaPy.benchmarks import Griewank
from NiaPy.benchmarks.utility import Task
from NiaPy.algorithms.basic import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.mke import MkeSolution

class MyBenchmark(object):
	def __init__(self):
		self.Lower = -5.12
		self.Upper = 5.12

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class MkeSolutionTestCase(TestCase):
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-2, 2, self.D), Task(self.D, 230, None, MyBenchmark())
		self.sol1, self.sol2, self.sol3 = MkeSolution(x=self.x), MkeSolution(self.task), MkeSolution(x=self.x)

	def test_x_fine(self):
		self.assertTrue(not False in (self.x == self.sol1.x))

	def test_f_fine(self):
		self.assertAlmostEqual(self.sol2.f, task.eval(self.sol2.x))
		self.assertEquals(self.sol1.f, inf)

	def test_uPersonalBest_fine(self):
		self.sol2.uPersonalBest()
		self.assertTrue(not False in (self.sol2.x == self.sol2.x_pb))
		self.assertEquals(self.sol2.f_pb, self.sol2.f)
		self.sol3.evaluate(self.task)
		self.sol3.x = full(self.task.D, -5.11)
		self.sol3.evaluate(self.task)
		self.sol3.uPersonalBest()
		self.assertTrue(False in (self.sol3.x == self.sol3.x_pb))
		self.assertEquals(self.sol2.f_pb, self.task.eval(self.x))

	def test_len_fine(self):
		self.assertEquals(len(self.sol1), len(self.x))
		self.assertEquals(len(self.sol2), self.D)

	def test_getitem_fine(self):
		for r in range(self.D): self.assertEquals(self.sol1[r], self.x[r])

	def test_generate_solution_fine(self):
		self.assertTrue(task.isFeasible(self.sol2))

	def test_eq_fine(self):
		self.assertFalse(self.sol1 == self.sol2)
		self.assertTrue(self.sol1 == self.sol1)

class MKEv1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.mkev1_custom = MonkeyKingEvolutionV1(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.mkev1_griewank = MonkeyKingEvolutionV1(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.mkev1_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.mkev1_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class MKEv2TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.mkev2_custom = MonkeyKingEvolutionV2(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.mkev2_griewank = MonkeyKingEvolutionV2(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.mkev2_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.mkev2_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class MKEv3TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.mkev3_custom = MonkeyKingEvolutionV3(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.mkev3_griewank = MonkeyKingEvolutionV3(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.mkev3_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.mkev3_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
