# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class, line-too-long
from unittest import TestCase
from numpy import array_equal
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.algorithms.basic import EvolutionStrategy1p1, EvolutionStrategyMp1, EvolutionStrategyMpL, EvolutionStrategyML, CovarianceMaatrixAdaptionEvolutionStrategy

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

class ES1p1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.es_custom = EvolutionStrategy1p1(D=self.D, nFES=1000, k=10, c_a=1.5, c_r=0.42, benchmark=MyBenchmark(), seed=1)
		self.es_customc = EvolutionStrategy1p1(D=self.D, nFES=1000, k=10, c_a=1.5, c_r=0.42, benchmark=MyBenchmark(), seed=1)
		self.es_griewank = EvolutionStrategy1p1(D=self.D, nFES=1000, k=15, c_a=1.2, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es_griewankc = EvolutionStrategy1p1(D=self.D, nFES=1000, k=15, c_a=1.2, c_r=0.5, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.es_custom.run()
		self.assertTrue(x)
		y = self.es_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.es_griewank.run()
		self.assertTrue(x)
		y = self.es_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

class ESMp1TestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.es_custom = EvolutionStrategyMp1(D=self.D, nFES=1000, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_customc = EvolutionStrategyMp1(D=self.D, nFES=1000, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_griewank = EvolutionStrategyMp1(D=self.D, nFES=1000, mu=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es_griewankc = EvolutionStrategyMp1(D=self.D, nFES=1000, mu=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.es_custom.run()
		self.assertTrue(x)
		y = self.es_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.es_griewank.run()
		self.assertTrue(x)
		y = self.es_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

class ESMpLTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.es_custom = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_customc = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=45, lam=55, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es1_custom = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es1_customc = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=55, lam=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_griewank = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es_griewankc = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=30, lam=50, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es1_griewank = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es1_griewankc = EvolutionStrategyMpL(D=self.D, nFES=1000, mu=50, lam=30, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.es_custom.run()
		self.assertTrue(x)
		y = self.es_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.es_griewank.run()
		self.assertTrue(x)
		y = self.es_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_custom1_works_fine(self):
		x = self.es1_custom.run()
		self.assertTrue(x)
		y = self.es1_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank1_works_fine(self):
		x = self.es1_griewank.run()
		self.assertTrue(x)
		y = self.es1_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

class ESMLTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.es_custom = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_customc = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es1_custom = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es1_customc = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, lam=35, k=50, c_a=1.1, c_r=0.5, benchmark=MyBenchmark(), seed=1)
		self.es_griewank = EvolutionStrategyML(D=self.D, nFES=1000, mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es_griewankc = EvolutionStrategyML(D=self.D, nFES=1000, mu=35, lam=45, k=45, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es1_griewank = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)
		self.es1_griewankc = EvolutionStrategyML(D=self.D, nFES=1000, mu=45, lam=35, k=25, c_a=1.5, c_r=0.5, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.es_custom.run()
		self.assertTrue(x)
		y = self.es_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.es_griewank.run()
		self.assertTrue(x)
		y = self.es_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_custom1_works_fine(self):
		x = self.es1_custom.run()
		self.assertTrue(x)
		y = self.es1_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank1_works_fine(self):
		x = self.es1_griewank.run()
		self.assertTrue(x)
		y = self.es1_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

class CMAESTestCase(TestCase):
	def setUp(self):
		self.D = 40
		self.es_custom = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=MyBenchmark(), seed=1)
		self.es_customc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=MyBenchmark(), seed=1)
		self.es1_custom = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=MyBenchmark(), seed=1)
		self.es1_customc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=MyBenchmark(), seed=1)
		self.es_griewank = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=Griewank(), seed=1)
		self.es1_customc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=MyBenchmark(), seed=1)
		self.es1_griewank = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=Griewank(), seed=1)
		self.es1_griewankc = CovarianceMaatrixAdaptionEvolutionStrategy(D=self.D, nFES=1000, benchmark=Griewank(), seed=1)

	def test_custom_works_fine(self):
		x = self.es_custom.run()
		self.assertTrue(x)
		y = self.es_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank_works_fine(self):
		x = self.es_griewank.run()
		self.assertTrue(x)
		y = self.es_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_custom1_works_fine(self):
		x = self.es1_custom.run()
		self.assertTrue(x)
		y = self.es1_custom.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

	def test_griewank1_works_fine(self):
		x = self.es1_griewank.run()
		self.assertTrue(x)
		y = self.es1_griewank.run()
		self.assertTrue(y)
		self.assertTrue(array_equal(x[0], y[0]))
		self.assertEqual(x[1], y[1])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
