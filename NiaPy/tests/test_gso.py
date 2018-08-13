# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import GlowwormSwarmOptimization, GlowwormSwarmOptimizationV1, GlowwormSwarmOptimizationV2, GlowwormSwarmOptimizationV3

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

class GSOTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.gso_custom = GlowwormSwarmOptimization(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark())
		self.gso_griewank = GlowwormSwarmOptimization(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.gso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.gso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class GSOv1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.gso_custom = GlowwormSwarmOptimizationV1(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark())
		self.gso_griewank = GlowwormSwarmOptimizationV1(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.gso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.gso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class GSOv2TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.gso_custom = GlowwormSwarmOptimizationV2(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark())
		self.gso_griewank = GlowwormSwarmOptimizationV2(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.gso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.gso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class GSOv3TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.gso_custom = GlowwormSwarmOptimizationV3(D=self.D, nFES=1000, NP=35, a=7, Rmin=0.1, Rmax=3, benchmark=MyBenchmark())
		self.gso_griewank = GlowwormSwarmOptimizationV3(D=self.D, nFES=1000, NP=10, a=5, Rmin=0.01, Rmax=3, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.gso_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.gso_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
