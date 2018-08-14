# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from unittest import TestCase
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm, FireworksAlgorithm, EnhancedFireworksAlgorithm, DynamicFireworksAlgorithm, DynamicFireworksAlgorithmGauss

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

class BBFWATestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.bbfwa_custom = BareBonesFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.bbfwa_griewank = BareBonesFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.bbfwa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.bbfwa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class FWATestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.fwa_custom = FireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.fwa_griewank = FireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.fwa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.fwa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class EFWATestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.efwa_custom = EnhancedFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.efwa_griewank = EnhancedFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.efwa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.efwa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class DFWATestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.dfwa_custom = DynamicFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.dfwa_griewank = DynamicFireworksAlgorithm(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.dfwa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.dfwa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

class DFWAGTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.dfwa_custom = DynamicFireworksAlgorithmGauss(D=self.D, nFES=1000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.dfwa_griewank = DynamicFireworksAlgorithmGauss(D=self.D, nFES=1000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self):
		fun = MyBenchmark().function()
		x = self.dfwa_custom.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

	def test_griewank_works_fine(self):
		fun = Griewank().function()
		x = self.dfwa_griewank.run()
		self.assertTrue(x)
		self.assertAlmostEqual(fun(self.D, x[0]), x[1], delta=1e2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
