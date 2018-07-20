# encoding=utf8
# pylint: disable=mixed-indentation
from unittest import TestCase
from NiaPy.benchmark.griewank import Griewank
from NiaPy.algorithms.basic import BareBonesFireworksAlgorithm

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

class CSTestCase(TestCase):
	def setUp(self):
		self.ca_custom = BareBonesFireworksAlgorithm(D=40, nFES=100000, n=10, C_a=2, C_r=0.5, benchmark=MyBenchmark())
		self.ca_griewank = BareBonesFireworksAlgorithm(D=40, nFES=100000, n=10, C_a=5, C_r=0.5, benchmark=Griewank())

	def test_custom_works_fine(self): 
		fun = MyBenchmark().function
		x = ca_custom.run()
		self.assertTrue(x)
		self.assertEqual(fun(self.D, x[0]), x[1])

	def test_griewank_works_fine(self):
		fun = Griewank().function
		x = ca_griewank.run()
		self.assertTrue(x)
		self.assertEqual(fun(self.D, x[0]), x[1])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
