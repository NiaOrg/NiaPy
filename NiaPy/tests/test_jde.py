# encoding=utf8
# pylint: disable=line-too-long, mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from numpy import random as rnd
from NiaPy.benchmarks.utility import Task
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolutionAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm
from NiaPy.algorithms.modified.jde import SolutionjDE


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

class SolutionjDETestCase(TestCase):
	def setUp(self):
		self.D, self.F, self.CR = 10, 0.9, 0.3
		self.x, self.task = rnd.uniform(10, 50, self.D), Task(self.D, 230, None, MyBenchmark())
		self.s1, self.s2 = SolutionjDE(task=self.task), SolutionjDE(x=self.x, CR=self.CR, F=self.F)

	def test_F_fine(self):
		self.assertAlmostEqual(self.s1.F, 2)
		self.assertAlmostEqual(self.s2.F, self.F)

	def test_cr_fine(self):
		self.assertAlmostEqual(self.s1.CR, 0.5)
		self.assertAlmostEqual(self.s2.CR, self.CR)

class jDETestCase(TestCase):
	def setUp(self):
		self.jde_custom = SelfAdaptiveDifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, benchmark=MyBenchmark())
		self.jde_griewank = SelfAdaptiveDifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, benchmark='griewank')

	def test_custom_works_fine(self):
		self.assertTrue(self.jde_custom.run())

	def test_griewank_works_fine(self):
		self.assertTrue(self.jde_griewank.run())

class dyNPjDETestCase(TestCase):
	def setUp(self):
		self.dynnpjde_custom = DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, benchmark=MyBenchmark())
		self.dynnpjde_griewank = DynNPSelfAdaptiveDifferentialEvolutionAlgorithm(D=10, NP=40, nFES=1000, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.1, Tao2=0.45, benchmark='griewank')

	def test_custom_works_fine(self):
		self.assertTrue(self.dynnpjde_custom.run())

	def test_griewank_works_fine(self):
		self.assertTrue(self.dynnpjde_griewank.run())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
