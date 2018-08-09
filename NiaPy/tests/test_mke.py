# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, too-many-function-args, old-style-class
from unittest import TestCase
from numpy import random as rnd, full, array_equal
from NiaPy.benchmarks import Griewank
from NiaPy.benchmarks.utility import Task
from NiaPy.algorithms.basic import MonkeyKingEvolutionV1, MonkeyKingEvolutionV2, MonkeyKingEvolutionV3
from NiaPy.algorithms.basic.mke import MkeSolution

class MyBenchmark:
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
		self.sol1, self.sol2, self.sol3 = MkeSolution(x=self.x), MkeSolution(task=self.task), MkeSolution(x=self.x)

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
